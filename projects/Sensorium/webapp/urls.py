"""
URL configuration for dashboard app.
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/mice/<str:mouse_id>/videos/', views.api_get_videos, name='api_get_videos'),
    path('api/mice/<str:mouse_id>/videos/<str:video_id>/info/', views.api_get_video_info, name='api_get_video_info'),
    path('api/mice/<str:mouse_id>/videos/<str:video_id>/neurons/', views.api_get_neurons, name='api_get_neurons'),
    path('api/mice/<str:mouse_id>/videos/<str:video_id>/plot/<str:data_type>/', views.api_get_plot_data, name='api_get_plot_data'),
    path('api/mice/<str:mouse_id>/cell_coordinates/', views.api_get_cell_coordinates, name='api_get_cell_coordinates'),
    path('api/mice/<str:mouse_id>/videos/<str:video_id>/video/', views.api_get_video_base64, name='api_get_video_base64'),
]
