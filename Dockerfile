FROM python:3.8-bullseye as build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH=/usr/bin/poetry/bin:$PATH \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
    POETRY_HOME=/usr/bin/poetry

# Python version must be 3.5 or higher
# Poetry must version be 1.1.7 or higher
RUN curl -sSL https://install.python-poetry.org > ./install-poetry.py && \
    python ./install-poetry.py && \
    rm ./install-poetry.py

# Create virtualenv for deployment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies
COPY poetry.lock poetry.lock
COPY pyproject.toml pyproject.toml

RUN poetry install --only main


FROM python:3.8-slim-bullseye

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV MPLCONFIGDIR=/tmp


WORKDIR /app/churn
COPY --from=build /opt/venv /opt/venv
COPY churn_library.py churn_library.py

ENTRYPOINT ["python", "churn_library.py"]
