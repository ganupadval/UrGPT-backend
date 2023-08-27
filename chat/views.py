# import requests
from .serializers import ChatMessageSerializer, ConversationSerializer
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import authentication_classes, permission_classes, api_view
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import JSONParser
from django.http import JsonResponse
import warnings
from typing import List
from .models import ChatMessage, Conversation
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import HuggingFacePipeline
from auto_gptq import AutoGPTQForCausalLM
from transformers import (
    AutoTokenizer,
    pipeline,
    logging,
)
from langchain.chains import ConversationChain
from langchain import HuggingFacePipeline

warnings.filterwarnings("ignore", category=UserWarning)


# Defining model configurations

model_name_or_path = "TheBloke/WizardLM-13B-V1.1-GPTQ"
model_basename = "wizardlm-13b-v1.1-GPTQ-4bit-128g.no-act.order"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


# quantizing model for resource-constrained environments
model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                                           model_basename=model_basename,
                                           use_safetensors=True,
                                           trust_remote_code=True,
                                           device="cuda:0",
                                           use_triton=use_triton,
                                           quantize_config=None)

logging.set_verbosity(logging.CRITICAL)

# defining model pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.2,
    top_k=50,
    top_p=0.95
)

llm = HuggingFacePipeline(pipeline=pipe)

memory = ConversationBufferMemory()

# API for title generator
API_URL = "https://api-inference.huggingface.co/models/czearing/article-title-generator"
headers = {"Authorization": f"Bearer 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"}

def generate_title(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]['generated_text']


# retriving the last 4 conversations from the db for memory eficiency while providing context to model
def retrieve_conversation(title, user):
    # number of conversations
    num_recent_conversations = 4

    # Retrieve the most recent conversation history from the database
    conversation_obj = Conversation.objects.get(title=title, user=user)
    conversation_id = getattr(conversation_obj, 'id')
    
    # Retrieve recent conversation messages
    conversation_context = ChatMessage.objects.filter(
        conversation_id=conversation_id
    ).order_by('-timestamp')[:num_recent_conversations:-1]
    
    # Storing the retrived data from db to model memory 
    lst = []
    for msg in conversation_context:
        input_msg = getattr(msg, 'user_response')
        output_msg = getattr(msg, 'ai_response')
        lst.append({"input": input_msg, "output": output_msg})
    
    for x in lst:
        inputs = {"input": x["input"]}
        outputs = {"output": x["output"]}
        memory.save_context(inputs, outputs)
    
   
    retrieved_chat_history = ChatMessageHistory(
        messages=memory.chat_memory.messages
    )

    return retrieved_chat_history


# Function to store the conversation to DB
def store_message(user_response, ai_response, conversation_id):
    ChatMessage.objects.create(
        user_response=user_response,
        ai_response=ai_response,
        conversation_id=conversation_id,
    )

# Function to create a Conversation in DB
def store_title(title, user):
    Conversation.objects.create(
        title=title,
        user=user
    )

# Function to Get all Chat history of conversation and create a new chat
@csrf_exempt
@api_view(['POST', 'GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def chat(request):
    if request.method == 'GET':
        request_data = JSONParser().parse(request)
        provided_title = request_data.get('title')
        user = request.user
        if provided_title:
            conversation_title = Conversation.objects.get(
                title=provided_title, user=user)
            conversation_id = getattr(conversation_title, 'id')
            ChatObj = ChatMessage.objects.filter(
                conversation_id=conversation_id).order_by('timestamp')
            Chat = ChatMessageSerializer(ChatObj, many=True)
            return JsonResponse(Chat.data, safe=False)
        else:
            return JsonResponse({'error': 'Title not provided'}, status=400)

    elif request.method == 'POST':
        request_data = JSONParser().parse(request)
        prompt = request_data.get('prompt')
        user = request.user
        provided_title = request_data.get('title')
        if provided_title:
            # Create a ChatMessageHistory instance
            retrieved_chat_history = retrieve_conversation(
                provided_title, user)

        else:
            memory.clear()
            retrieved_chat_history = ChatMessageHistory(messages=[])

        reloaded_chain = ConversationChain(
            llm=llm,
            memory=ConversationBufferMemory(
                chat_memory=retrieved_chat_history),
            verbose=True
        )

        response = reloaded_chain.predict(input=prompt)

        if provided_title:
            title = provided_title
        else:
            # Generate a default title if not provided
            title = generate_title({
                "inputs": response
            })
            store_title(title, user)

        conversation_title = Conversation.objects.get(title=title, user=user)
        conversation_id = getattr(conversation_title, 'id')
        store_message(prompt, response, conversation_id)

        return JsonResponse({
            'ai_responce': response,
            'title':title
        }, status=201)



# Retriving all conversations of a user ( Titles only )

@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])  
def get_title(request):
    user=request.user
    titles= Conversation.objects.filter(user=user)
    serialized= ConversationSerializer(titles, many=True)
    return JsonResponse(serialized.data, safe=False)

# Delete a conversation by providing title of conversation
@csrf_exempt   
@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated]) 
def delete_conversation(request):
    user=request.user
    data= JSONParser().parse(request)
    title= data.get('title')
    obj=Conversation.objects.get(user=user, title=title)
    obj.delete()
    return JsonResponse("Deleted succesfully", safe=False)

@csrf_exempt   
@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def get_data(request):
    request_data = JSONParser().parse(request)
    provided_title = request_data.get('title')
    user = request.user
    if provided_title:
        conversation_title = Conversation.objects.get(
            title=provided_title, user=user)
        conversation_id = getattr(conversation_title, 'id')
        ChatObj = ChatMessage.objects.filter(
            conversation_id=conversation_id).order_by('timestamp')
        Chat = ChatMessageSerializer(ChatObj, many=True)
        return JsonResponse(Chat.data, safe=False)
    else:
        return JsonResponse({'error': 'Title not provided'}, status=400)
