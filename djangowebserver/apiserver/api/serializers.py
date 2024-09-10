from rest_framework import serializers

class TransformerSerializer(serializers.Serializer):
    aminoAcidSeq = serializers.CharField(required=True, allow_blank=False, max_length=500)

