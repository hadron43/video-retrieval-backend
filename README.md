# FastAPI App

## Vector Engine installation
Install vector engine:
```shell
git submodule init
git submodule update
```

## Model files initialization
Download folder from [google drive](https://drive.google.com/drive/folders/1-zgjeLheefUQyG6jlOAJ6HjQ70BaveMr?usp=share_link) and add the folder in project root directory. (You should have model files inside `fintuned_model` directory.)

## Setup

1. Create a virtual environment named `.venv`:
```shell
python3 -m venv .venv
```

2. Activate the virtual environment:
```shell
source .venv/bin/activate
```

3. Install dependencies:
```shell
pip install -r requirements.txt
```

## Running the Development Server

1. Start the development server using `uvicorn`:
```
uvicorn main:app --reload
```

This will start the server on `http://127.0.0.1:8000`.
