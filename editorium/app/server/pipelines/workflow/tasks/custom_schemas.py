import base64
from PIL import Image
from io import BytesIO
import base64

from marshmallow import Schema, fields, ValidationError


class ImageField(fields.Field):
    def _deserialize(self, value, attr, data):
        if not value:
            return None
        if self.context.get("from_api", False):
            if type(value) is str:
                return Image.open(BytesIO(base64.b64decode(value)))
            else:
                raise ValidationError("Image must be a base64 string")
        return value
        
    
    def _serialize(self, value, attr, obj):
        if not value:
            return None
        if self.context.get("from_api", False):
            buffered = BytesIO()
            value.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue())
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
