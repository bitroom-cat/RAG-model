from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
import ollama

def index(request):
    # This serves the HTML page
    return render(request, 'index.html')

def home(request):
    # This is the status check your URL file is looking for
    return JsonResponse({"status": "API is running"})

@api_view(['POST'])
def chat_with_boost(request):
    user_query = request.data.get('message')
    
    # Calls your local RTX 3050 via Ollama
    response = ollama.chat(model='llama3', messages=[
        {'role': 'user', 'content': user_query}
    ])
    
    ai_answer = response['message']['content']
    return Response({"agent_response": ai_answer})