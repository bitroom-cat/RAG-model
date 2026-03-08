from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),               # http://127.0.0.1:8000/api/
    path('status/', views.home, name='home'),          # http://127.0.0.1:8000/api/status/
    path('chat/', views.chat_with_boost, name='chat'), # http://127.0.0.1:8000/api/chat/
]