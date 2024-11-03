import base64
from PIL import Image
from io import BytesIO
import base64

from mashmallow import Schema, fields, ValueError


class ImageField(fields.Field):
    def _deserialize(self, value, attr, data):
        if not value:
            return None
        if self.context.get("from_api", False):
            if type(value) is str:
                return Image.open(BytesIO(base64.b64decode(value)))
            else:
                raise ValueError("Image must be a base64 string")
        return value
        
    
    def _serialize(self, value, attr, obj):
        if not value:
            return None
        if self.context.get("from_api", False):
            buffered = BytesIO()
            value.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue())
        return value


class InputOutputSchema(Schema):
    images = fields.List(ImageField(), required=False)
    masks = fields.List(ImageField(), required=False)
    texts = fields.List(fields.Str(), required=False)
    data = fields.list(fields.Dict(), required=False)
    