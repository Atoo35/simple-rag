import httpx
import constants


def generate_embed(input):
    # Define the request payload
    payload = {
        "model": "nomic-embed-text",
        "input": input
    }

    # Send the request
    response = httpx.post(f"{constants.BASE_URL}/embeddings", json=payload)
    return (response.json())["data"]
