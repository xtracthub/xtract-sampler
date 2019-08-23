
import boto3
import pickle as pkl
import json

sqs = boto3.resource('sqs')
sqs2 = boto3.client('sqs')

def put_on_queue(msg):
    queue = sqs.get_queue_by_name(QueueName='xtract-crawl-queue')

    str_msg = json.dumps(msg)

    response = queue.send_message(MessageBody=str_msg)
    return response

def pull_off_queue():
    response = sqs2.receive_message(
    QueueUrl='https://sqs.us-east-1.amazonaws.com/576668000072/xtract-crawl-queue',
    AttributeNames=[
        'SentTimestamp'
    ],
    MaxNumberOfMessages=1,
    MessageAttributeNames=[
        'All'
    ],
    VisibilityTimeout=0,
    WaitTimeSeconds=0)

    if "Messages" in response:
        message = response["Messages"][0]
        body = message["Body"]
        receipt_handle = message['ReceiptHandle']

        print(body)
        print(receipt_handle)

        sqs2.delete_message(
            QueueUrl='https://sqs.us-east-1.amazonaws.com/576668000072/xtract-crawl-queue',
            ReceiptHandle=receipt_handle)     
        return json.loads(body)

    else:
        print("Messages unavailable!")
        return None


put_on_queue({"hi": "hello"})
pull_off_queue()
