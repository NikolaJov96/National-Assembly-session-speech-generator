import json
import os

from datetime import datetime


SESSIONS_FOLDER = 'sessions/'
SESSIONS_FILE = SESSIONS_FOLDER + 'sessions_data.json'
SESSIONS_TEXT_FOLDER = SESSIONS_FOLDER + 'sessions_text/'


def session_data_to_text(session, out_file):
    '''
    Writes the speeches of a single session into provided file
    '''

    for speech in session['speeches']:
        out_file.write('{}\n\n'.format(speech['speaker']))
        out_file.write('{}\n\n'.format(speech['speech']))
    out_file.write('\n\n')


def filter_unwanted_speeches(sessions_data):
    '''
    Filters the following:
    - Speeches shorter than 200 characters
    - Speeches containing the word 'amandman' more than 2 times (indicaton dull speech)
    '''

    for session in sessions_data:
        session['speeches'] = list(filter(
            lambda speech: len(speech['speech']) > 200 and speech['speech'].lower().count('amandman') < 2,
            session['speeches']))
    return sessions_data


def filter_unwanted_sessions(sessions_data):
    '''
    Filters session with less than 2 speeches
    '''

    return list(filter(lambda session: len(session['speeches']) > 1, sessions_data))


def sessions_data_to_text(sessions_data):
    '''
    Applies filters to the provided data and stores it in the desired format for training
    '''

    print('Original num of speeches: {}'.format(sum(map(lambda session: len(session['speeches']), sessions_data))))
    print('Original num of sessions: {}'.format(len(sessions_data)))

    sessions_data = filter_unwanted_speeches(sessions_data)
    sessions_data = filter_unwanted_sessions(sessions_data)

    print('Num of speeches after filtering: {}'.format(sum(map(lambda session: len(session['speeches']), sessions_data))))
    print('Num of sessions after filtering: {}'.format(len(sessions_data)))

    sessions_data = sorted(sessions_data, key=lambda session: datetime.strptime(session['date'], '%d.%m.%Y'))

    if not os.path.exists(SESSIONS_TEXT_FOLDER):
        os.makedirs(SESSIONS_TEXT_FOLDER)

    for i, session in enumerate(sessions_data):
        print('Exporting session {}/{}\r'.format(i + 1, len(sessions_data)), end='')
        with open('{}session_{}.txt'.format(SESSIONS_TEXT_FOLDER, i), 'w') as out_file:
            session_data_to_text(session, out_file)
    print()


def main():
    '''
    Text preparation steps:
    - Load scraped data
    - Filter unwanted data
    - Save in training-friendly format
    '''

    if not os.path.exists(SESSIONS_FOLDER) or not os.path.exists(SESSIONS_FILE):
        print('Scraped data not found')
        return

    with open(SESSIONS_FILE, 'r') as in_file:
        sessions_data = json.load(in_file)
        sessions_data_to_text(sessions_data)


if __name__ == '__main__':
    main()
