import base64
from PIL import Image
import base64

from marshmallow import Schema, fields, ValidationError
from pipelines.common.utils import pil_as_dict, pil_from_dict


class ImageField(fields.Field):
    def _deserialize(self, value, attr, data):
        if not value:
            return None
        if self.context.get("from_api", False):
            if type(value) is dict:
                return  pil_from_dict(value)
            else:
                raise ValidationError("Image must be a dictionary with 'data', 'width', 'height' and 'mode' keys.")
        return value
        
    
    def _serialize(self, value, attr, obj):
        if not value:
            return None
        if self.context.get("from_api", False):
            return pil_as_dict(value)
        return value
    

class BoxField(fields.Field):
    def _deserialize(self, value, attr, data):
        if not value:
            return None
        if self.context.get("from_api", False):
            if type(value) is dict:
                return  [value['x'], value['y'], value['x2'], value['y2']]
            else:
                raise ValidationError("Image must be a dictionary with 'x', 'y', 'x2' and 'y2' keys.")
        return value
        
    
    def _serialize(self, value, attr, obj):
        if not value:
            return None
        if self.context.get("from_api", False):
            return {'x': value[0], 'y': value[1], 'x2': value[2], 'y2': value[3]}
        return value


class InputOutputSchema(Schema):
    images = fields.List(ImageField(), required=False, load_default=[])
    boxes = fields.List(BoxField(), required=False, load_default=[])
    texts = fields.List(fields.Str(), required=False, load_default=[])
    data = fields.Dict(required=False, load_default={})
    data_list = fields.List(fields.Dict(), required=False, load_default=[])
