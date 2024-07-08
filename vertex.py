"""
This file contains the Vertex class, which is used to represent a vertex in the graph.
"""
__author__ = "Zihan Wang"
class Vertex:
    def __init__(self,id):
        self._id = id
        self._varyings = {}

    def attach_varying(self, v_name, v_value):
        self._varyings[v_name] = v_value

    def get_varying(self, v_name):
        return self._varyings[v_name]

    def get_id(self):
        return self._id