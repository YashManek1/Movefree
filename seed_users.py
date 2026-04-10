import requests
import sys

API_KEY = 'AIzaSyCHWisx4Ozokv-sEJ8-GfJEtUzog4SzBvI'
DB_URL  = 'https://moovefree-ab842-default-rtdb.asia-southeast1.firebasedatabase.app'

USERS = [
    {
        'email':       'blind@moovefree.app',
        'password':    'Moove@123',
        'displayName': 'Alex (Blind User)',
        'role':        'visually_impaired',
    },
    {
        'email':       'caretaker@moovefree.app',
        'password':    'Moove@123',
        'displayName': 'Sarah (Caretaker)',
        'role':        'caretaker',
    },
]

uids   = {}
tokens = {}

print('Seeding MooveFree demo users...\n')

for u in USERS:
    print('-> ' + u['role'] + ' : ' + u['email'])
    r = requests.post(
        'https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=' + API_KEY,
        json={
            'email':             u['email'],
            'password':          u['password'],
            'displayName':       u['displayName'],
            'returnSecureToken': True,
        },
        timeout=10,
    )
    d = r.json()

    if 'error' in d:
        msg = d['error'].get('message', '')
        if msg == 'EMAIL_EXISTS':
            print('   Already exists, signing in...')
            r2 = requests.post(
                'https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=' + API_KEY,
                json={
                    'email':             u['email'],
                    'password':          u['password'],
                    'returnSecureToken': True,
                },
                timeout=10,
            )
            d = r2.json()
            if 'error' in d:
                print('   FAIL: ' + d['error'].get('message', ''))
                continue
        else:
            print('   FAIL: ' + msg)
            continue

    uid = d['localId']
    tok = d['idToken']
    uids[u['role']]   = uid
    tokens[u['role']] = tok

    pr = requests.put(
        DB_URL + '/users/' + uid + '.json?auth=' + tok,
        json={
            'email':       u['email'].lower().strip(),
            'role':        u['role'],
            'displayName': u['displayName'],
            'createdAt':   0,
        },
        timeout=10,
    )
    status = 'OK' if pr.status_code == 200 else 'FAIL(' + str(pr.status_code) + ')'
    print('   ' + status + '  UID: ' + uid)

if 'caretaker' in uids and 'visually_impaired' in uids:
    print('\n-> Binding caretaker to blind user...')
    br = requests.put(
        DB_URL + '/bindings/' + uids['caretaker'] + '.json?auth=' + tokens['caretaker'],
        json={
            'patientUid':  uids['visually_impaired'],
            'patientName': 'Alex (Blind User)',
        },
        timeout=10,
    )
    if br.status_code == 200:
        print('   BOUND OK')
    else:
        print('   BIND FAIL: ' + br.text)

    blind_uid = uids['visually_impaired']
    print('\nBLIND_USER_UID=' + blind_uid)
    print('\nAdd the above UID to moovefree_indoor/.env and moovefree_outdoor/.env\n')

print('-------------------------------------------')
print('Demo login credentials:')
print()
print('  BLIND USER (Phone 2 - visually impaired):')
print('    Email   : blind@moovefree.app')
print('    Password: Moove@123')
print()
print('  CARETAKER (Phone 1 / Dashboard):')
print('    Email   : caretaker@moovefree.app')
print('    Password: Moove@123')
print()
print('  Both accounts are pre-linked.')
print('-------------------------------------------')
