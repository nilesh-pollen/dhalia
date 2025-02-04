from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from pathlib import Path
import os
from anthropic import Anthropic
import json
import numpy as np
import math
from enum import Enum
from typing_extensions import Annotated
import joblib
import requests

app = FastAPI(title="Pollen Dhalia API")


def validate_claude_response(content: str) -> Dict[str, Any]:
    """
    Validates and formats Claude's response to ensure it's valid JSON.

    Args:
        content (str): Raw response from Claude

    Returns:
        Dict[str, Any]: Parsed JSON response

    Raises:
        ValueError: If response cannot be parsed as valid JSON
    """
    try:
        # Try to parse as JSON directly first
        return json.loads(content)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from markdown or other formatting
        # Look for JSON between triple backticks if present
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                json_str = content[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in markdown: {str(e)}")

        raise ValueError("Could not extract valid JSON from Claude's response")


def get_infisical_secret(
    secret_name,
    workspace_id="4b2ecbaa-4f0f-4ac7-82a7-b4f311a320fb",
    environment="development",
):
    url = f"https://us.infisical.com/api/v3/secrets/raw"

    # Get the token from environment variable instead of file
    token = os.getenv("INFISICAL_TOKEN")
    if not token:
        raise RuntimeError("INFISICAL_TOKEN environment variable not set")

    headers = {"Authorization": f"Bearer {token}"}
    params = {"workspaceId": workspace_id, "environment": environment}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch secrets: {response.text}")

    secrets = response.json()["secrets"]
    for secret in secrets:
        if secret["secretKey"] == secret_name:
            return secret["secretValue"]

    raise RuntimeError(
        f"Secret {secret_name} not found in environment {environment}"
    )


try:
    # Get Claude API key from Infisical
    claude_api_key = get_infisical_secret(
        secret_name="anthropic-key", environment="dev"
    )

    # Get prompt from Infisical
    claude_prompt = get_infisical_secret(
        "dhalia-anthropic-prompt", environment="prod"
    )

except Exception as e:
    raise RuntimeError(f"Failed to fetch secrets from Infisical: {str(e)}")

anthropic = Anthropic(api_key=claude_api_key)

# Load pickle files
try:
    data_dir = Path("data")
    repeated_inv = joblib.load(data_dir / "repeated_inv.pkl")
    prompt_1 = joblib.load(data_dir / "prompt_1.pkl")
    prompt_7 = joblib.load(data_dir / "prompt_7.pkl")
    quantile_in_days_to_shelf_life_range = joblib.load(
        data_dir / "quantile_in_days_to_shelf_life_range.pkl"
    )
    model_prompt_1 = joblib.load(data_dir / "model_prompt_1.pkl")
    model_prompt_7 = joblib.load(data_dir / "model_prompt_7.pkl")
except FileNotFoundError as e:
    raise RuntimeError(f"Required pickle file not found: {e.filename}")


# Pydantic models
class AdditionalContext(BaseModel):
    product_category: Optional[str] = None
    product_subcategory: Optional[str] = None
    seller_name: Optional[str] = None
    shelf_life_remaining_days: Optional[int] = None
    brand_name: Optional[str] = None


class NaturalLanguageRequest(BaseModel):
    query: str
    context: Optional[AdditionalContext] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "How many days will it take for Zwitsal products with shelf life 40-60 days to reach 50-60% depletion?",
                    "context": {
                        "product_category": "Baby Care",
                        "product_subcategory": "Baby Bath",
                        "seller_name": "unilever_indonesia",
                    },
                },
                {
                    "query": "For SKU ZTH03178 with shelf life range 40-60, when will it reach 90-100% depletion?",
                    "context": {
                        "product_category": "Personal Care",
                        "product_subcategory": "Hair Care",
                        "seller_name": "unilever_indonesia",
                        "brand_name": "zwitsal",
                        "shelf_life_remaining_days": 45,
                    },
                },
            ]
        }
    }


# Business Logic Layer
class PromptLogic:
    @staticmethod
    def get_shelf_life_range(input_shelf_life: int) -> str:
        for i in quantile_in_days_to_shelf_life_range.keys():
            if i < input_shelf_life:
                continue
            return quantile_in_days_to_shelf_life_range[i]
        return quantile_in_days_to_shelf_life_range[i]

    @staticmethod
    def get_depletion_range(input_depletion_percent: int) -> str:
        return f"{input_depletion_percent // 10 * 10}_{((input_depletion_percent // 10) + 1) * 10}"

    @staticmethod
    def predict_prompt_1(
        brand: str,
        product_category: str,
        product_subcategory: str,
        seller_name: str,
        shelf_life_range: str,
        depletion_percent: int,
    ) -> float:
        test_data = [
            brand,
            product_category,
            product_subcategory,
            seller_name,
            shelf_life_range,
            depletion_percent,
        ]
        return math.ceil(model_prompt_1.predict(test_data))

    @staticmethod
    def get_prompt_2_percentage(
        shelf_life_range: str,
        warehouse_country: str,
        brand: str,
        min_depletion: float,
    ) -> float:
        prompt_2_A = repeated_inv[
            (repeated_inv.shelf_life_range == shelf_life_range)
            & (repeated_inv.warehouse_country == warehouse_country)
            & (repeated_inv.brand == brand)
            & (repeated_inv.depletion >= min_depletion)
        ]

        prompt_2_B = repeated_inv[
            (repeated_inv.shelf_life_range == shelf_life_range)
            & (repeated_inv.warehouse_country == warehouse_country)
            & (repeated_inv.brand == brand)
        ]

        if prompt_2_B.shape[0] == 0:
            raise ValueError(
                f"No data found for brand {brand} in {warehouse_country}"
            )

        ans = (prompt_2_A.shape[0] / prompt_2_B.shape[0]) * 100

        if ans <= 5:
            ans = float(np.random.uniform(low=5, high=10, size=(1,))[0])
        if ans >= 95:
            ans = float(np.random.uniform(low=90, high=95, size=(1,))[0])

        return ans

    @staticmethod
    def get_prompt_4_categories(
        shelf_life_range: str,
        warehouse_country: str,
        min_depletion: float,
        max_time: float,
    ) -> List[str]:
        filtered_data = repeated_inv[
            (repeated_inv.shelf_life_range == shelf_life_range)
            & (repeated_inv.warehouse_country == warehouse_country)
            & (repeated_inv.depletion > min_depletion)
            & (repeated_inv.time < max_time)
        ]
        return list(filtered_data.product_category.unique())

    @staticmethod
    def get_prompt_5_categories(
        seller_name: str,
        priority: str,
        warehouse_country: str,
        min_depletion: float,
        max_time: float,
    ) -> List[str]:
        filtered_data = repeated_inv[
            (repeated_inv.seller_name == seller_name)
            & (repeated_inv.priority == priority)
            & (repeated_inv.warehouse_country == warehouse_country)
            & (repeated_inv.depletion > min_depletion)
            & (repeated_inv.time < max_time)
        ]
        return list(filtered_data.product_category.unique())

    @staticmethod
    def get_prompt_6_subcategories(
        seller_name: str,
        priority: str,
        warehouse_country: str,
        min_depletion: float,
        max_time: float,
    ) -> List[str]:
        filtered_data = repeated_inv[
            (repeated_inv.seller_name == seller_name)
            & (repeated_inv.priority == priority)
            & (repeated_inv.warehouse_country == warehouse_country)
            & (repeated_inv.depletion > min_depletion)
            & (repeated_inv.time < max_time)
        ]
        return list(filtered_data.product_subcategory.unique())

    @staticmethod
    def predict_prompt_7(
        sku_number: str,
        brand: str,
        product_category: str,
        product_subcategory: str,
        seller_name: str,
        shelf_life_remaining_days: int,
        depletion_percent: int,
    ) -> float:
        test_data = [
            sku_number,
            brand,
            product_category,
            product_subcategory,
            seller_name,
            shelf_life_remaining_days,
            depletion_percent,
        ]
        return math.ceil(model_prompt_7.predict(test_data))

    @staticmethod
    def get_prompt_8_percentage(
        shelf_life_range: str,
        warehouse_country: str,
        sku_number: str,
        min_depletion: float,
    ) -> float:
        prompt_8_A = repeated_inv[
            (repeated_inv.shelf_life_range == shelf_life_range)
            & (repeated_inv.warehouse_country == warehouse_country)
            & (repeated_inv.sku_number == sku_number)
            & (repeated_inv.depletion >= min_depletion)
        ]

        prompt_8_B = repeated_inv[
            (repeated_inv.shelf_life_range == shelf_life_range)
            & (repeated_inv.warehouse_country == warehouse_country)
            & (repeated_inv.sku_number == sku_number)
        ]

        if prompt_8_B.shape[0] == 0:
            raise ValueError(
                f"No data found for SKU {sku_number} in {warehouse_country}"
            )

        ans = (prompt_8_A.shape[0] / prompt_8_B.shape[0]) * 100

        if ans <= 5:
            ans = float(np.random.uniform(low=5, high=10, size=(1,))[0])
        if ans >= 95:
            ans = float(np.random.uniform(low=90, high=95, size=(1,))[0])

        return ans


# Natural Language Processing Layer
class NLProcessor:
    def __init__(self, anthropic_client, system_prompt):
        self.client = anthropic_client
        self.system_prompt = system_prompt

    def validate_claude_response(self, content: str) -> Dict[str, Any]:
        """
        Validates and formats Claude's response to ensure it's valid JSON.

        Args:
            content (str): Raw response from Claude

        Returns:
            Dict[str, Any]: Parsed JSON response

        Raises:
            ValueError: If response cannot be parsed as valid JSON
        """
        try:
            # Try to parse as JSON directly first
            return json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from markdown or other formatting
            # Look for JSON between triple backticks if present
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end != -1:
                    json_str = content[start:end].strip()
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in markdown: {str(e)}")

            raise ValueError(
                "Could not extract valid JSON from Claude's response"
            )

    def validate_context(
        self, result: Dict[str, Any], context: AdditionalContext
    ) -> None:
        if result["endpoint"] == "prompt_1":
            required_fields = {
                "product_category": context.product_category,
                "product_subcategory": context.product_subcategory,
                "seller_name": context.seller_name,
            }
            missing_fields = [
                k for k, v in required_fields.items() if v is None
            ]
            if missing_fields:
                raise ValueError(
                    f"Missing required context for prompt 1: {', '.join(missing_fields)}"
                )
        elif result["endpoint"] == "prompt_7":
            required_fields = {
                "product_category": context.product_category,
                "product_subcategory": context.product_subcategory,
                "seller_name": context.seller_name,
                "brand_name": context.brand_name,
                "shelf_life_remaining_days": context.shelf_life_remaining_days,
            }
            missing_fields = [
                k for k, v in required_fields.items() if v is None
            ]
            if missing_fields:
                raise ValueError(
                    f"Missing required context for prompt 7: {', '.join(missing_fields)}"
                )

    async def process_query(
        self, query: str, context: AdditionalContext
    ) -> Dict[str, Any]:
        try:
            # Make the API call to Claude
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0,
                system=self.system_prompt,
                messages=[{"role": "user", "content": query}],
            )

            # Get the response text
            content = message.content[0].text

            # Log the raw response for debugging
            print(f"Raw Claude response: {content}")

            # Use the validation function to parse the response
            try:
                result = self.validate_claude_response(content)
            except ValueError as e:
                print(f"Failed to parse Claude response: {str(e)}")
                raise

            # Validate and update context for specific endpoints
            if result["endpoint"] in ["prompt_1", "prompt_7"]:
                self.validate_context(result, context)

                if result["endpoint"] == "prompt_1":
                    result["parameters"].update(
                        {
                            "product_category": context.product_category,
                            "product_subcategory": context.product_subcategory,
                            "seller_name": context.seller_name,
                        }
                    )
                elif result["endpoint"] == "prompt_7":
                    result["parameters"].update(
                        {
                            "brand": context.brand_name,
                            "product_category": context.product_category,
                            "product_subcategory": context.product_subcategory,
                            "seller_name": context.seller_name,
                            "shelf_life_remaining_days": context.shelf_life_remaining_days,
                        }
                    )

            return result

        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            print(f"Query: {query}")
            print(f"Context: {context}")
            raise ValueError(f"Error processing query: {str(e)}")


# API Layer setup
logic = PromptLogic()
nl_processor = NLProcessor(anthropic, claude_prompt)

# Endpoints ordered by prompt number


@app.post("/pollen/dhalia/natural_language_query")
async def natural_language_query(request: NaturalLanguageRequest):
    try:
        result = await nl_processor.process_query(
            request.query, request.context or AdditionalContext()
        )

        if result["endpoint"] == "prompt_1":
            params = result["parameters"]
            time = logic.predict_prompt_1(
                params["brand"],
                params["product_category"],
                params["product_subcategory"],
                params["seller_name"],
                params["shelf_life_range"],
                params["depletion_percent"],
            )
            return {
                "result": {"time": time},
                "explanation": result["explanation"],
                "interpreted_as": result,
            }

        elif result["endpoint"] == "prompt_7":
            params = result["parameters"]
            time = logic.predict_prompt_7(
                params["sku_number"],
                params["brand"],
                params["product_category"],
                params["product_subcategory"],
                params["seller_name"],
                params["shelf_life_remaining_days"],
                params["depletion_percent"],
            )
            return {
                "result": {"time": time},
                "explanation": result["explanation"],
                "interpreted_as": result,
            }

        elif result["endpoint"] == "prompt_2":
            params = result["parameters"]
            percentage = logic.get_prompt_2_percentage(
                params["shelf_life_range"],
                params["warehouse_country"],
                params["brand"],
                params["min_depletion"],
            )
            return {
                "result": {"percentage": percentage},
                "explanation": result["explanation"],
                "interpreted_as": result,
            }

        elif result["endpoint"] == "prompt_4":
            params = result["parameters"]
            categories = logic.get_prompt_4_categories(
                params["shelf_life_range"],
                params["warehouse_country"],
                params["min_depletion"],
                params["max_time"],
            )
            return {
                "result": {"product_categories": categories},
                "explanation": result["explanation"],
                "interpreted_as": result,
            }

        elif result["endpoint"] == "prompt_5":
            params = result["parameters"]
            categories = logic.get_prompt_5_categories(
                params["seller_name"],
                params["priority"],
                params["warehouse_country"],
                params["min_depletion"],
                params["max_time"],
            )
            return {
                "result": {"product_categories": categories},
                "explanation": result["explanation"],
                "interpreted_as": result,
            }

        elif result["endpoint"] == "prompt_6":
            params = result["parameters"]
            subcategories = logic.get_prompt_6_subcategories(
                params["seller_name"],
                params["priority"],
                params["warehouse_country"],
                params["min_depletion"],
                params["max_time"],
            )
            return {
                "result": {"product_subcategories": subcategories},
                "explanation": result["explanation"],
                "interpreted_as": result,
            }

        elif result["endpoint"] == "prompt_8":
            params = result["parameters"]
            percentage = logic.get_prompt_8_percentage(
                params["shelf_life_range"],
                params["warehouse_country"],
                params["sku_number"],
                params["min_depletion"],
            )
            return {
                "result": {"percentage": percentage},
                "explanation": result["explanation"],
                "interpreted_as": result,
            }

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported endpoint: {result['endpoint']}",
            )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your query: {str(e)}",
        )


# Prompt 1 endpoints
@app.get(
    "/pollen/dhalia/stats/prompt1/time_to_depletion_by_brand", deprecated=True
)
async def get_prompt_1_stats(
    brand: str = "zwitsal",
    shelf_life_range: str = "40_60",
    depletion_range: str = "50_60",
):
    try:
        time = prompt_1[
            (prompt_1.brand == brand)
            & (prompt_1.shelf_life_range == shelf_life_range)
            & (prompt_1.depletion_range == depletion_range)
        ].time.values[0]
        return {"time": time}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/pollen/dhalia/ml/prompt1/time_to_depletion_ml")
async def get_prompt_1_ml(
    brand: str,
    product_category: str,
    product_subcategory: str,
    seller_name: str,
    shelf_life_range: str,
    depletion_percent: int,
):
    try:
        time = logic.predict_prompt_1(
            brand,
            product_category,
            product_subcategory,
            seller_name,
            shelf_life_range,
            depletion_percent,
        )
        return {"time": time}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Prompt 2 endpoint
@app.get("/pollen/dhalia/stats/prompt2/brand_depletion_percentage")
async def get_prompt_2(
    shelf_life_range: str = "40_60",
    warehouse_country: str = "indonesia",
    brand: str = "pepsodent",
    min_depletion: float = 50,
):
    try:
        percentage = logic.get_prompt_2_percentage(
            shelf_life_range, warehouse_country, brand, min_depletion
        )
        return {"percentage": percentage}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Prompt 4 endpoint
@app.get("/pollen/dhalia/stats/prompt4/category_depletion")
async def get_prompt_4(
    shelf_life_range: str = "40_60",
    warehouse_country: str = "indonesia",
    min_depletion: float = 50,
    max_time: float = 30,
):
    try:
        categories = logic.get_prompt_4_categories(
            shelf_life_range, warehouse_country, min_depletion, max_time
        )
        return {"product_categories": categories}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Prompt 5 endpoint
@app.get("/pollen/dhalia/stats/prompt5/seller_categories")
async def get_prompt_5(
    seller_name: str = "unilever_indonesia",
    priority: str = "p2",
    warehouse_country: str = "indonesia",
    min_depletion: float = 50,
    max_time: float = 30,
):
    try:
        categories = logic.get_prompt_5_categories(
            seller_name, priority, warehouse_country, min_depletion, max_time
        )
        return {"product_categories": categories}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Prompt 6 endpoint
@app.get("/pollen/dhalia/stats/prompt6/seller_subcategories")
async def get_prompt_6(
    seller_name: str = "unilever_indonesia",
    priority: str = "p2",
    warehouse_country: str = "indonesia",
    min_depletion: float = 50,
    max_time: float = 30,
):
    try:
        subcategories = logic.get_prompt_6_subcategories(
            seller_name, priority, warehouse_country, min_depletion, max_time
        )
        return {"product_subcategories": subcategories}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Prompt 7 endpoints
@app.get(
    "/pollen/dhalia/stats/prompt7/time_to_depletion_by_sku", deprecated=True
)
async def get_prompt_7_stats(
    sku_number: str = "ZTH03178",
    shelf_life_range: str = "40_60",
    depletion_range: str = "90_100",
):
    try:
        time = prompt_7[
            (prompt_7.sku_number == sku_number)
            & (prompt_7.shelf_life_range == shelf_life_range)
            & (prompt_7.depletion_range == depletion_range)
        ].time.values[0]
        return {"time": time}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/pollen/dhalia/ml/prompt7/sku_depletion_time_ml")
async def get_prompt_7_ml(
    sku_number: str,
    brand: str,
    product_category: str,
    product_subcategory: str,
    seller_name: str,
    shelf_life_remaining_days: int,
    depletion_percent: int,
):
    try:
        time = logic.predict_prompt_7(
            sku_number,
            brand,
            product_category,
            product_subcategory,
            seller_name,
            shelf_life_remaining_days,
            depletion_percent,
        )
        return {"time": time}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Prompt 8 endpoint
@app.get("/pollen/dhalia/stats/prompt8/sku_depletion_percentage")
async def get_prompt_8(
    shelf_life_range: str = "40_60",
    warehouse_country: str = "indonesia",
    sku_number: str = "67451987",
    min_depletion: float = 10,
):
    try:
        percentage = logic.get_prompt_8_percentage(
            shelf_life_range, warehouse_country, sku_number, min_depletion
        )
        return {"percentage": percentage}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",  # Replace 'main' with your Python file name without .py
        host="0.0.0.0",
        port=8001,
        reload=True,
        workers=1,
    )
