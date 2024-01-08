"""
Designed to send email notifications, abbr. EMN
based on gmail services and google oauth2 credentials
"""
import base64
import copy
import os.path
import time
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import cv2
from func_timeout import func_set_timeout
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


class Gmail:
    def __init__(self, manual_token_path):
        self.manual_token_path = manual_token_path
        # check manual_token_path is available
        if os.path.exists(manual_token_path):
            self.auto_token_path = os.path.join(os.path.dirname(manual_token_path), 'auto_token.json')
        else:
            raise FileExistsError(f'{manual_token_path} does not exist')

    def send(self, message):
        if message is not None:
            # get credentials
            self._get_credentials()
            # create message
            to = message['to']
            subject = message['subject']
            text = ''.join(message['text'].split('    '))
            image_in_text = message['image_in_text']
            attachment = message['attachment']
            message = self._create_message(to, subject, text)
            message = self._attach_with_image_in_text(message, image_in_text)
            message = self._attach_with_attachment(message, attachment)

            # send message
            self._send_message({'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()})

    @staticmethod
    def _create_message(to, subject, text):
        _message = MIMEMultipart()
        _message['to'] = to
        _message['subject'] = subject
        _message.attach(MIMEText(text))
        return _message

    @staticmethod
    def _attach_with_image_in_text(_message, image_path):
        image_path = [image_path] if type(image_path) is not list else image_path
        for idx, img_p in enumerate(image_path):
            html = f"""
            <html>
              <body>
                <img src="cid:image{idx}">
              </body>
            </html>
            """
            # attach the image
            with open(img_p, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-ID', f'<image{idx}>')

            _message.attach(MIMEText(html, 'html'))
            _message.attach(img)
        return _message

    @staticmethod
    def _attach_with_attachment(_message, file_path):
        file_path = [file_path] if type(file_path) is not list else file_path
        # add the image attachment
        for f_p in file_path:
            with open(f_p, 'rb') as f:
                attachment = MIMEApplication(f.read())
                attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(f_p))
            _message.attach(attachment)
        return _message

    @func_set_timeout(60)
    def _get_credentials(self):
        SCOPES = ['https://mail.google.com/']

        def _manual_auth(_manual_token_path, _SCOPES):
            _flow = InstalledAppFlow.from_client_secrets_file(_manual_token_path, _SCOPES)
            _credentials = _flow.run_local_server(port=0)
            self._log('access granted')
            return _credentials

        # try to get credentials if there is existing auto_token
        if os.path.exists(self.auto_token_path):
            self._log('auto_token exists')
            credentials = Credentials.from_authorized_user_file(self.auto_token_path, SCOPES)
            # if invalid, try auto refresh
            if not credentials.valid:
                try:
                    credentials.refresh(Request())
                    self._log('credentials auto refresh')
                except:
                    # get user manual authorization
                    self._log('auto refresh failed, need manual authorization')
                    credentials = _manual_auth(self.manual_token_path, SCOPES)
        else:
            # get user manual authorization
            self._log('no existing auto_token, need manual authorization')
            credentials = _manual_auth(self.manual_token_path, SCOPES)

        # save new token for next
        with open(self.auto_token_path, 'w') as t:
            t.write(credentials.to_json())
        self._log('new auto_token is saved')
        self.gmail_service = build('gmail', 'v1', credentials=credentials)

    def _send_message(self, _message):
        _message = (self.gmail_service.users().messages().send(userId='me', body=_message).execute())
        self._log(f'sent message to {_message}')

    def _log(self, txt):  # print with device name
        print(f'[{self.__class__.__name__}]\t{txt}')

    def __del__(self):
        self._log('Closed.')


class EmailNotifier(Gmail):
    def __init__(self, run_flag, shared_param_dict, **kwargs_CFG):
        """
        get shared values and queues
        """
        self.run_flag = run_flag
        # shared params
        self.autosave_flag = shared_param_dict['autosave_flag']
        self.email_image = shared_param_dict['email_image']  # sent from save_center

        """
        pass config static parameters
        """
        """ module own config """
        EMN_CFG = kwargs_CFG['EMAIL_NOTIFIER_CFG']
        try:
            self.message_sys_start = EMN_CFG['message_sys_start']
            self.message_obj_detected = EMN_CFG['message_obj_detected']
            self.message_daily_check = EMN_CFG['message_daily_check']
            self.message_daily_check_send_time = EMN_CFG['message_daily_check_send_time']
        except:
            self.message_sys_start = None
            self.message_obj_detected = None
            self.message_daily_check = None
            self.message_daily_check_send_time = []

        """
        self content
        """
        self.message_sys_start_send_flag = 0
        self.cache_image_path = './cache.jpg'
        self.message_obj_detected_send_flag = True
        self.message_daily_check_send_flag = True

        self._log('Start...')

        """
        inherit father class __init__ para
        """
        super().__init__(EMN_CFG['manual_token_path'])

    def run(self):
        while self.run_flag.value:
            # send sys start message with a 15 sec delay
            delay = 15  # set delay for waiting other modules
            if self.message_sys_start_send_flag == delay and self.message_sys_start is not None:
                self.send(self.message_sys_start)
                self.message_sys_start_send_flag = False
            elif type(self.message_sys_start_send_flag) is int:
                self.message_sys_start_send_flag += 1

            # send object detection message
            if self.autosave_flag.value and self.message_obj_detected_send_flag and self.message_obj_detected is not None:
                time.sleep(2)
                # saved cache image
                cv2.imwrite(self.cache_image_path, self.email_image.value)
                # send the email
                message = copy.deepcopy(self.message_obj_detected)
                message['image_in_text'].append(self.cache_image_path)
                self.send(message)
                os.remove(self.cache_image_path)  # clear the cache image
                self.message_obj_detected_send_flag = False
            # prevent duplicate sending
            elif not self.autosave_flag.value:
                self.message_obj_detected_send_flag = True

            # send daily status message
            if time.strftime('%H-%M', time.localtime()) in self.message_daily_check_send_time and self.message_daily_check_send_flag:
                self.send(self.message_daily_check)
                self.message_daily_check_send_flag = False
            # prevent duplicate sending
            elif time.strftime('%H-%M', time.localtime()) not in self.message_daily_check_send_time:
                self.message_daily_check_send_flag = True

            # check once per second
            time.sleep(1)

    def __del__(self):
        self._log(f"Closed. Timestamp: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")


if __name__ == '__main__':
    mes = {
        'to'           : '1740781310szc@gmail.com',
        'subject'      : 'Test Email from Gmail API',
        'text'         :
            """
                Hello, this is a test email 
                sent using the Gmail API.
            """,
        'image_in_text': [],
        'attachment'   : [],
    }
    Gmail('./email_notifier_token/manual_token.json').send(mes)
    print('Email sent successfully')
