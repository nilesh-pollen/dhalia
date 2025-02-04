FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy the entire project including data directory
COPY . .

# Set the PATH to include uv
ENV PATH="/root/.local/bin:$PATH"

# Expose port
EXPOSE 8001

# Command to run the application using uv
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]