import hashlib

def hash_cspace(c_space):
    c_space_bytes = c_space.tobytes()
    hash_object = hashlib.md5(c_space_bytes)
    return hash_object.hexdigest()
