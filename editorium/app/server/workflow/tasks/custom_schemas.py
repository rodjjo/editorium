import base64
from PIL import Image
import base64

from marshmallow import Schema, fields, ValidationError


def pil_as_dict(pil_image):
    return {
        'data': base64.b64encode(pil_image.tobytes()).decode('utf-8'),
        'width': int(pil_image.width),
        'height': int(pil_image.height),
        'mode': str(pil_image.mode),
    }



def pil_from_dict(data):
    return Image.frombytes(data['mode'], (data['width'], data['height']),  base64.b64decode(data['data']))


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

class BoxSchema(Schema):
    x = fields.Int(required=True)
    y = fields.Int(required=True)
    x2 = fields.Int(required=True)
    y2 = fields.Int(required=True)

class InputOutputSchema(Schema):
    images = fields.List(ImageField(), required=False, load_default=[])
    boxes = fields.List(fields.Nested(BoxSchema), required=False, load_default=[])
    texts = fields.List(fields.Str(), required=False, load_default=[])
    data = fields.Dict(required=False, load_default={})
    data_list = fields.List(fields.Dict(), required=False, load_default=[])
