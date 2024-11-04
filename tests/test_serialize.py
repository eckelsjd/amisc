"""Test serialization mixin classes."""
from amisc.serialize import Base64Serializable, PickleSerializable, Serializable, StringSerializable, YamlSerializable


class ClassTestBase64(Base64Serializable):
    def __init__(self, value):
        self.value = value


def test_base64_serializable():
    obj = ClassTestBase64(42)
    serialized = obj.serialize()
    deserialized = ClassTestBase64.deserialize(serialized)
    assert obj.value == deserialized.value


class ClassTestString(StringSerializable):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f'TestClass({self.value})'


def test_string_serializable():
    obj = ClassTestString(42)
    serialized = obj.serialize()
    deserialized = ClassTestString.deserialize(serialized)
    assert obj.value == deserialized.value


class ClassTestPickle(PickleSerializable):
    def __init__(self, value):
        self.value = value


def test_pickle_serializable(tmp_path):
    fname = tmp_path / 'test_pickle.pkl'
    obj = ClassTestPickle(42)
    serialized = obj.serialize(fname)  # noqa: F841
    deserialized = ClassTestPickle.deserialize(fname)
    assert obj.value == deserialized.value


class ClassTestYaml(Serializable):
    def __init__(self, value):
        self.value = value

    def serialize(self):
        return {'value': self.value}

    @classmethod
    def deserialize(cls, serialized_data):
        return cls(serialized_data['value'])


def test_yaml_serializable():
    meta_obj = YamlSerializable(obj=ClassTestYaml)
    serialized = meta_obj.serialize()
    deserialized = YamlSerializable.deserialize(serialized)
    assert meta_obj.obj == deserialized.obj
