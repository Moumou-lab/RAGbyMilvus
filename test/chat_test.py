import requests

url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "model": "Qwen/Qwen3-8B",
    "max_tokens": 512,
    "enable_thinking": False,
    "thinking_budget": 4096,
    "min_p": 0.05,
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1,
    "messages": [
        {
            "content": "你是师大先生",
            "role": "system"
        },
        {
            "content": "你好",
            "role": "user"
        }
    ]
}
headers = {
    "Authorization": "Bearer sk-nhnklopejonbklumkchlnsjaluxbetocvqdzevgcrjptjlpj",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json())