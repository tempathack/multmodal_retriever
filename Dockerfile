FROM python:3.10-slim

# Set the working directory in the container
WORKDIR .

# Copy the current directory contents into the container at /app
COPY configs /configs
COPY dbs /dbs
COPY file_tracking /file_tracking
COPY preprocessed_data /preprocessed_data
COPY raw_data /raw_data
COPY .streamlit /.streamlit
COPY src /src
COPY utils /utils
COPY .env /.env
COPY st_db.py /st_db.py
COPY save.pkl /save.pkl
COPY poetry.lock /poetry.lock
COPY pyproject.toml /pyproject.toml
# Install any needed packages specified in requirements.txt
RUN pip install poetry

RUN poetry config virtualenvs.create false && poetry install --no-dev
# Make port 80 available to the world outside this container
EXPOSE 8501

# Define environment variable (optional)
ENV NAME FastAPI-Docker

# Run Streamlit app when the container launches
CMD ["streamlit", "run", "st_db.py"]