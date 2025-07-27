import requests

url = "https://api.siliconflow.cn/v1/embeddings"

payload = {
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": "Nihao",
    "encoding_format": "float"
}
headers = {
    "Authorization": "Bearer sk-nhnklopejonbklumkchlnsjaluxbetocvqdzevgcrjptjlpj",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json())