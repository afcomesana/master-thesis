from django.shortcuts import render, get_object_or_404, redirect
from .models import Question, Answer, Participant
import wfdb
import os
import random
from pathlib import Path
import logging
import json

from thesis.settings import ECG_DIR, ORDERED_LEADS
    
logger = logging.getLogger(__name__)

# Create your views here.
def index(request):
    first_question = Question.objects.order_by('id').first()
    n_questions = Question.objects.count()
    return render(request, 'home.html', {
        'first_question_id': first_question.id,
        'n_questions': n_questions
        })
    
def question(request, question_id):

    # Initialize context with question
    question = get_object_or_404(Question, id=question_id)
    
    context = {
        'question': question,
        'form_data': request.POST,
        'questions_range': [i+1 for i in range(Question.objects.count())],
    }
    
    # Create or set participant
    if 'p_id' not in request.session:
        participant = Participant.objects.create()
        request.session['p_id'] = participant.id
    else:
        participant = get_object_or_404(Participant, id=request.session['p_id'])
        

    # Save answer:
    if request.method == 'POST':
        answer_text = request.POST.get('answer', '').strip()
        
        if answer_text:
            print(answer_text)
            # Save answer
            try:
                Answer.objects.update_or_create(
                    participant=participant,
                    question=question,
                    defaults={
                        'answer_text': answer_text
                    }
                )
            except Exception as e:
                logger.error(
                    f'Failed to save answer for participant {participant.id} '
                    f'Question {question.id}: {e}'
                )
                return redirect('error')
            
            # Redirect to next question or thank you page
            next_question = question.next()
            
            if next_question:
                return redirect('question', question_id = next_question.id)

            else:
                return redirect('thank_you')

    if question.ecg_record:
        
        # Read raw ECG data
        record = wfdb.rdrecord(ECG_DIR / question.ecg_record.filename)
        
        # Reorder leads to match how they are rendered att Uppsala Akademiska Sjukhuset
        leads_order = [record.sig_name.index(lead_name) for lead_name in ORDERED_LEADS]

        # Add ECG data to the context
        context['signals'] = json.dumps(record.p_signal[:,leads_order].T.tolist())
        context['fs'] = record.fs
        context['lead_names'] = [record.sig_name[idx] for idx in leads_order]
        context['age'] = question.ecg_record.age
        context['sex'] = 'Male' if question.ecg_record.sex else 'Female'


    return render(request, 'question.html', context)


def thank_you(request):
    request.session.flush()
    return render(request, 'thank_you.html')

def error(request):
    return render(request, 'error.html')