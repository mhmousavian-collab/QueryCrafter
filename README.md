# Query Creafter
A local agent, building and executing sql queries based on user prompt. With a minimal UI.

## Setup
To setup rename the `*.gguf` model file to `model.gguf`. Then copy the `model.gguf` file to main directory of project.
We used `sqlcoder-7b-2.Q4_K_M.gguf` found in [here](https://huggingface.co/MaziyarPanahi/sqlcoder-7b-2-GGUF/blob/main/sqlcoder-7b-2.Q4_K_M.gguf). we recommend using the same.

After providing the model file you can run the project like 
```
$ docker compose up --build -d
```

The UI will be exposed at `localhost:8000`.

## GPU availability
We built and tested this software on a machine with a **NVIDIA GeForce RTX 3050** GPU.
If you don't have access to an Nvidia GPU, comment reserved resources for `ollama` service in `docker-compose.yml`.

## Notes
Sometimes it takes a few seconds for the model to be loaded. Also models have some non-deterministic behaviour, rerunning queries might lead to answer.
 
