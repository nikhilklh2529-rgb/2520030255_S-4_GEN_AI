
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

class MongoLogger:
    def __init__(self):
        mongo_uri = os.getenv("MONGO_URI")

        if not mongo_uri:
            raise ValueError("MONGO_URI missing in .env")

        self.client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=10000
        )

        # Verify MongoDB connection
        self.client.admin.command("ping")

        self.db = self.client["llm_monitoring"]
        self.collection = self.db["call_logs"]

        print("MongoDB Atlas connected!")

    def log_call(self, log_data):
        log_data["timestamp"] = datetime.now(timezone.utc)
        self.collection.insert_one(log_data)
        print("Call details saved in MongoDB!")

    def close(self):
        self.client.close()
