# Run

```terminal
./requirements/setup_venv.sh mistral
source ~/python_venvs/mistral/bin/activate
cd mistral_chat

# (OPTIONAL) Get index and 
(mistral) python3 build_index.py

# launch server with fastapi
(mistral) uvicorn main:app --host 0.0.0.0 --port 8000

# open browser
http://0.0.0.0:8000/
```

# Example

<img src="img/chat_example.png" alt="drawing" width="1000"/>
