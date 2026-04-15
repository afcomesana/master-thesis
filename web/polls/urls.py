from django.urls import path

from . import views


urlpatterns = [
    path('', views.index, name='home'),
    path('question/<int:question_id>', views.question, name='question'),
    path('thank-you/', views.thank_you, name='thank_you'),
    path('error/', views.error, name='error'),
]