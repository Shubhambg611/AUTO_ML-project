from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .serializers import FileUploadSerializer
from .preprocessing import preprocess_data
from .evaluation import evaluate_models
from .gemini_integration import get_gemini_insights
from .report_generation import generate_report

class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_serializer = FileUploadSerializer(data=request.data)
        if file_serializer.is_valid():
            file = request.FILES['file']
            data = preprocess_data(file)
            results = evaluate_models(data)
            insights = get_gemini_insights(data)
            report = generate_report(results, insights)
            return Response(report, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
