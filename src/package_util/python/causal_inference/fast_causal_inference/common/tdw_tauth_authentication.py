from typing import Dict, Union
import requests
import base64
import binascii
import urllib.parse
import json
import pyDes
import socket
import uuid
import random
from datetime import datetime as dt
from .. import RAINBOW_CONF


class TdwTauthAuthentication:
    TDW_TAUTH = RAINBOW_CONF["tdw"]["tauth"]

    def __init__(self, userName: str, cmk: str, target: str, proxyUser: str = None) -> None:
        super().__init__()
        self.userName = userName
        self.cmk = cmk
        self.expire = True
        self.expireTimeStamp = int(dt.timestamp(dt.now())) * 1000
        self.proxyUser = proxyUser
        self.ip = self.get_host_ip()
        self.sequence = random.randint(0, 999)
        self.identifier = {
            "user": self.userName,
            "host": self.ip,
            "target": target,
            "lifetime": 7200000
        }
        self.ClientAuthenticator: Dict[str, Union[str, int]] = {
            "principle": self.userName,
            "host": self.ip
        }

    def get_host_ip(self) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        return ip

    def getSessionTicket(self):
        requestUrl = TdwTauthAuthentication.TDW_TAUTH
        self.identifier["timestamp"] = int(dt.timestamp(dt.now()))
        identifierBody = base64.b64encode(
            bytes(json.dumps(self.identifier), 'utf8'))
        response = requests.get(requestUrl, params={"ident": identifierBody})
        self.sessionTicket = response.text
        if response.status_code != 200:
            raise ValueError(
                "Request session ticket fail: {}".format(response.text))

    def decryptClientTicket(self):
        sessionTicket = json.loads(self.sessionTicket)
        self.serviceTicket = sessionTicket["st"]
        clientTicket = sessionTicket["ct"]
        cmkEpoch = sessionTicket["cmkEpoch"]
        clientTicket = base64.b64decode(clientTicket)
        try:
            cmk = bytes.fromhex(base64.b64decode(self.cmk).decode('utf8'))
            DESede = pyDes.triple_des(cmk, pyDes.ECB, padmode=pyDes.PAD_PKCS5)
            clientTicket = json.loads(
                str(DESede.decrypt(clientTicket), 'utf8'))
            self.expireTimeStamp = clientTicket["timestamp"] + \
                                   clientTicket["lifetime"] - 60000
            self.sessionKey = base64.b64decode(clientTicket['sessionKey'])
        except (binascii.Error, json.decoder.JSONDecodeError):
            raise ValueError(
                f"There is not proper master key. Please make sure you have download the key with ID {cmkEpoch}")

    def constructAuthentication(self) -> Dict[str, str]:
        self.ClientAuthenticator["timestamp"] = int(
            dt.timestamp(dt.now())) * 1000
        self.ClientAuthenticator["nonce"] = uuid.uuid1().hex
        self.ClientAuthenticator["sequence"] = self.sequence
        self.sequence += 1
        if self.proxyUser:
            self.ClientAuthenticator["proxyUser"] = self.proxyUser
        ClientAuthenticator = bytes(
            json.dumps(self.ClientAuthenticator), 'utf8')
        DESede = pyDes.triple_des(
            self.sessionKey, pyDes.ECB, padmode=pyDes.PAD_PKCS5)
        ClientAuthenticator = DESede.encrypt(ClientAuthenticator)
        authentication = "tauth." + self.serviceTicket + "." + \
                         str(base64.b64encode(ClientAuthenticator), 'utf8')
        if self.identifier['target'] == 'metadataservice':
            return {"tdw-sign": authentication}
        elif self.identifier['target'] == 'idex-openapi':
            return {"authentication": authentication}
        elif self.identifier['target'] == 'security_center':
            return {"secure-authentication": urllib.parse.quote(authentication)}
        else:
            return {"secure-authentication": authentication}

    def isExpire(self) -> bool:
        return self.expireTimeStamp <= int(dt.timestamp(dt.now())) * 1000

    def getAuthentication(self):
        if self.isExpire():
            self.getSessionTicket()
            self.decryptClientTicket()
        return self.constructAuthentication()
