"""

Request daa from DB and return them

"""
import json


class DataDistributor:

    def __init__(self, parent, testing, user_id, db_manager):
        self.parent=parent
        self.test_data = [
            {"id": 1, "age": 25, "name": "Alice", "city": "Berlin", "active": True, "score": 88.5},
            {"id": 2, "age": 31, "name": "Bob", "city": "Munich", "active": False, "score": 72.0},
            {"id": 3, "age": 29, "name": "Charlie", "city": "Hamburg", "active": True, "score": 95.3},
            {"id": 4, "age": 40, "name": "Diana", "city": "Cologne", "active": True, "score": 81.7},
            {"id": 5, "age": 22, "name": "Eve", "city": "Stuttgart", "active": False, "score": 67.2}
        ]
        self.testing = testing

        self.send_type = "env_data"

        self.user_id=user_id

        self.db_manager=db_manager


    async def send_data(self, data={}):
        env_id = data.get("env_id")
        if self.testing is False:
            # generate admin_data
            if env_id is not None:
                db_root = f"users/{self.user_id}/env/{env_id}/datastore/"
                raw_data:dict = self.db_manager.get_data(db_root)
                data = []
                for nid, attrs in raw_data.items():
                    if "id" not in attrs:
                        data.append(attrs.update({"id": nid}))
            else:
                data = []
        else:
            data = self.test_data
        await self.parent.send(
            text_data=json.dumps({
                "type": self.send_type,
                "message": "success",
                "admin_data": data
        }))
        print("Data sidtributed to foront")