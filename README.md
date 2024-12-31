rags are annoying, but I've made my second one: https://steveseguin.github.io/rag/

this requires ollama installed locally. it should be configured to allow CORS from https websites.
```
const OLLAMA_ENDPOINT = "http://localhost:11434";
```

Also note the following models need to be configured
```
const EMBEDDING_MODEL = "granite-embedding:30m"; // or your preferred model
const COMPLETION_MODEL = "llama3.2:latest";
```

it's pretrained to use a knowledge database for VDO.Ninja and will remotely load it. If you remove that though, it will give you the option to create and save a new knowledge database (embeddings).

Once loaded, it will persist in local storage on your browser.
