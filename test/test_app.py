"""
Test implementation of a Restful API
"""
from flask import Flask
from flask_restful import Api, Resource

CFG_PORT = 5000
app = Flask(__name__)
api = Api(app)


class HelloWorld(Resource):
    """
    Test API Endpoint
    """
    def get(self):
        """
        Getter for hello world
        """

        data={"data": "Hi Programming Project 2023!"}

        return data


api.add_resource(HelloWorld, '/hello')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=CFG_PORT)
