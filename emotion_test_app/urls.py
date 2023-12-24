from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('start_test/', views.start_test, name='start_test'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('download_results/', views.download_results, name='download_results'),
    path('stop_test/', views.stop_test, name='stop_test'),  # Add this line
]
