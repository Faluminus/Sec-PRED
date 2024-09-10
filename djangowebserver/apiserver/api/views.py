from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .mlmodels.transformer import TransformerPredict
from .serializers import TransformerSerializer
import json


@api_view(['POST','GET'])
def PredictWithTransformer(request):
    """Gets prediction using TransformerPredict module"""

    if request.method == 'GET':
        return Response(status=200)
    
    input_ = json.loads(request.data)
    serializer = TransformerSerializer(input_)
    try:
        serializer.is_valid()
        transformer = TransformerPredict(serializer.validated_data['aminoAcidSeq'])
        transformer.start()
        output = transformer.translate_to_secondary_structure()
        return Response(output,status=201)
    except:
        return Response(status=442)
    

