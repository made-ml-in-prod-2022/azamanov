import logging
import random
import sys

import click
import requests
from typing import Dict, Any

from model import Item

MIN_AGE = 20
MAX_AGE = 80
NUM_REQUESTS = 100

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def generate_json() -> Dict:
    item = Item(trestbps=160, chol=234, fbs=1, restecg=2,
                thalach=131, exang=0, oldpeak=0.1,
                age=random.randint(MIN_AGE, MAX_AGE),
                sex=random.randint(0, 1),
                cp=random.randint(0, 1),
                slope=random.randint(0, 1),
                ca=random.randint(0, 1),
                thal=random.randint(0, 1))
    return item.json()


@click.command()
@click.argument("url")
@click.argument("num_requests", type=int)
def request_command(url: str, num_requests: int) -> None:
    logging.info(f"Number of requests: {num_requests}")
    for i in range(num_requests):
        data = generate_json()
        logging.info(f"Data is generated: {data}")
        response = requests.post(url, data)
        logging.info(f"Response: {response.json()}")


if __name__ == "__main__":
    request_command()
