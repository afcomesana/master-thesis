from django.db import models
from django.core.validators import MaxValueValidator

# Create your models here.
class Participant(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Participant {self.id}'
    
class EcgRecord(models.Model):
    filename = models.CharField(max_length=100)
    
    # 1 = Male
    # 0 = Female
    sex = models.BooleanField()
    
    age = models.PositiveSmallIntegerField(validators=[MaxValueValidator(150)])
    
    def __str__(self):
        return f'{'Male' if self.sex else 'Female'} ({self.age})'
    
class Question(models.Model):
    question_text = models.CharField(max_length = 300)
    ecg_record = models.ForeignKey(EcgRecord, null=True, blank=True, on_delete=models.SET_NULL)

    def __str__(self):
        return f'({self.id}) {self.question_text}'
    
    def next(self):
        return Question.objects.filter(id__gt=self.id).first()

class Answer(models.Model):
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    answer_text = models.CharField(max_length=1000)
    
    
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['participant', 'question'], 
                name='unique_answer_per_participant_question'
            )
        ]
        
    def __str__(self):
        return f'Participant {self.participant.id} — {self.question.question_text}'

    