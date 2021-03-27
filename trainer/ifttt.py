"""
IFTTT Callbacks
"""
from requests import post
from tensorflow.keras.callbacks import Callback

class IFTTTTrainingCallback(Callback):
    ifttt_key = 'bxoTj4tM935zzKaQcJUQtO'
    event_name = None

    def __init__(self, jobname):
        self.jobname = jobname

    def logstring(self, logs):
        return ', '.join(
            f'{key}: {value:.02f}' 
            for key,value in logs.items()
        )

    def _call_ifttt_webhook(self, value1=None, value2=None, value3=None):
        print('Sending IFTTT', self.event_name, 'event')
        post(f'https://maker.ifttt.com/trigger/{self.event_name}/with/key/{self.ifttt_key}', 
            json={'value1': value1, 'value2': value2, 'value3': value3})


class IFTTTTrainingCompleteCallback(IFTTTTrainingCallback):
    event_name = 'training_complete'

    def on_train_end(self, logs):
        logstr = self.logstring(logs)
        self._call_ifttt_webhook(self.jobname, logstr)


class IFTTTTrainingProgressCallback(IFTTTTrainingCallback):
    event_name = 'training_progress'

    def __init__(self, jobname, total_epochs):
        super(IFTTTTrainingProgressCallback, self).__init__(jobname)
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs):
        logstr = self.logstring(logs)
        percentage = epoch / self.total_epochs * 100
        if percentage >= 10 and percentage % 10 == 0:
            self._call_ifttt_webhook(self.jobname, int(percentage), logstr)