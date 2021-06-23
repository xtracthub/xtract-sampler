
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

def put_on_results_queue(msg):
    queue = sqs.get_queue_by_name(QueueName='xtract-results-queue')

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
        # TODO: Uncomment when live. 
        sqs2.delete_message(
            QueueUrl='https://sqs.us-east-1.amazonaws.com/576668000072/xtract-crawl-queue',
            ReceiptHandle=receipt_handle)     
        return json.loads(body)

    else:
        print("Messages unavailable!")
        return None


def pull_off_results_queue():
    response = sqs2.receive_message(
    QueueUrl='https://sqs.us-east-1.amazonaws.com/576668000072/xtract-results-queue',
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
        # TODO: Uncomment when live. 
        sqs2.delete_message(
            QueueUrl='https://sqs.us-east-1.amazonaws.com/576668000072/xtract-results-queue',
            ReceiptHandle=receipt_handle)     
        return json.loads(body)

    else:
        print("Messages unavailable!")
        return None

#import csv
#import os
#searched_files = set()  # TODO: This state isn't saved during checkpointing.
#outfile = 'le-features.csv'
# message1 = pull_off_results_queue()
# {"file_path": "/projects/DLHub/tyler/sampler_train_set/48148-845.xyz", "file_size": 686, "sample_type": "tabular", "total_time": 0.19516873359680176}
#with open(outfile, 'a', newline='') as f:
#    csv_writer = csv.writer(f)
    #if os.path.getsize(outfile) == 0:
    #    csv_writer.writerow(["path", "size", "file_label", "infer_time"])
#    i = 0    
#    while True:
#        message1 = pull_off_results_queue()
#        if message1["file_path"] not in searched_files:
#            csv_writer.writerow([message1["file_path"], message1["file_size"], message1["sample_type"], message1["total_time"]])
#            print("Successfully written file attributes to disk!") 
#            searched_files.add(message1["file_path"])
#        else:
#            print("Duplicate file!")
#
        #if i == 5: 
        #    break
#        i+=1
#        print("Processed: {}".format(i))



# put_on_queue({"hey":"filet"})
