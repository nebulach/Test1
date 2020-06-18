from flask import Flask, render_template, request, redirect, url_for, make_response, json, send_file
from custom_tokenizer import FullBPETokenizer
from bert_serving.client import BertClient
import json
import numpy as np
from models import UtteranceResult
from models import SimilarUtteranceResult
from sklearn.metrics.pairwise import cosine_similarity
from enum import Enum
import torch
import pandas as pd
import pickle
import os
APP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(APP_PATH, 'template/')
HOME_PATH = os.path.join(TEMPLATE_PATH, 'template_html_startup/index.html')

class Tokenizers(Enum):
    PRETRAINED_256 = FullBPETokenizer(vocab_file='/data/nebulach/Datasets/BERT/Pretrained/256_8/vocab.txt',code_file='/data/nebulach/Datasets/BERT/Pretrained/256_8/code')

class BertClients(Enum):
    #PRETRAINED_256_REMAP_CLS = BertClient(port=5551, port_out=5552)
    PRETRAINED_256_REMAP_PRECLS = BertClient(port=5553, port_out=5554)
    PRETRAINED_256_EMB = BertClient(port=5555, port_out=5556)
    PRETRAINED_256_REMAP_PRECLS_NEW = BertClient(port=5580, port_out=5581)
    PRETRAINED_256_REMAP_PRECLS_NEW2 = BertClient(port=5590, port_out=5591)

app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

with open ('/data/nebulach/workspace/bert-as-a-service/web/user_utt_202001_ko_mobile_speaker_goal_remap_classify_match_ex_capsule.pkl', 'rb') as fp:
    user_utt = pickle.load(fp)
user_utt['match'] = user_utt['match'].apply(lambda x : 'T' if x==1 else 'F')
with open ('/data/nebulach/workspace/bert-as-a-service/web/user_utt_202001_ko_mobile_speaker_goal.pkl', 'rb') as fp:
    user_utt2 = pickle.load(fp)
with open ('/data/nebulach/workspace/bert-as-a-service/web/train_utt_202001_ko_mobile_speaker.pkl', 'rb') as fp:
    train_utt = pickle.load(fp)

user_utt2 = user_utt2.reset_index(drop=True)
train_utt = train_utt.reset_index(drop=True)

with open('/data/nebulach/workspace/bert-as-a-service/web/cluster_num_list.pkl', 'rb') as fp:
    cluster_num_list = pickle.load(fp)

with open('/data/nebulach/workspace/bert-as-a-service/web/capsule_evaluation_all.pkl', 'rb') as fp:
    capsule_evaluation_list = pickle.load(fp)

cluster_centers_decoded = np.load("/data/nebulach/workspace/bert-as-a-service/web/user_utt_cluster_centers.npy")
confusion_result = np.load("/data/nebulach/workspace/bert-as-a-service/web/confusion_result.npy")
confusion_result_log = np.load("/data/nebulach/workspace/bert-as-a-service/web/confusion_result_log.npy")

print(len(user_utt))
user_utt_encoded = np.load("/data/nebulach/workspace/bert-as-a-service/web/user_utt_202001_ko_mobile_speaker_encoded.npy")

#user_utt_encoded = user_utt_encoded[0:100000]

print(len(user_utt_encoded))

capsule_remap_list_new = ['bixby.astrologyResolver_KR', 'bixby.audioBookResolver_KR',
       'bixby.briefing', 'bixby.contentInfoResolver_KR',
       'bixby.currencyResolver_KR', 'bixby.deliveryTakeoutResolver_KR',
       'bixby.deviceControlResolver_KR', 'bixby.emailResolver_KR',
       'bixby.galaxyBuds', 'bixby.howToUseBixby',
       'bixby.languageLearningResolver_KR', 'bixby.launcher',
       'bixby.launcherResolver_KR', 'bixby.mapResolver_KR',
       'bixby.mediaResolver_KR', 'bixby.movieResolver_KR',
       'bixby.musicResolver_KR', 'bixby.newsResolver_KR',
       'bixby.podcastResolver_KR', 'bixby.radioResolver_KR',
       'bixby.searchResolver_KR', 'bixby.settingsApp',
       'bixby.sportsScoreResolver_KR', 'bixby.stockInfoResolver_KR',
       'bixby.systemResolver_KR', 'bixby.translatorsResolver_KR',
       'bixby.tvChannelResolver_KR', 'bixby.tvShowsResolver_KR',
       'bixby.volumeResolver_KR', 'bixby.weatherInfoResolver_KR',
       'viv.accessibilityApp', 'viv.arEmojiApp', 'viv.arZoneApp',
       'viv.bixbyChat_entertainment_koKR', 'viv.bixbyChat_koKR',
       'viv.bixbyHomeApp', 'viv.bixbyQAKR_Entity', 'viv.bixbyQAKR_Geo',
       'viv.bixbyQAKR_Google', 'viv.bixbyQAKR_Person', 'viv.bixbyVisionApp',
       'viv.bixbyVoiceApp', 'viv.calculator', 'viv.calendarApp',
       'viv.cameraApp', 'viv.clockApp', 'viv.contactApp', 'viv.deviceFAQ',
       'viv.deviceMaintenanceApp', 'viv.devicePromotion', 'viv.flightStats',
       'viv.galaxyApps', 'viv.galleryApp', 'viv.gracenote',
       'viv.interparkFlight', 'viv.interparkHotel', 'viv.lottery',
       'viv.mangoPlate', 'viv.manrecipe', 'viv.messageApp', 'viv.myFilesApp',
       'viv.noActionSocialNetwork', 'viv.phoneApp', 'viv.reminderApp',
       'viv.samsungAccountApp', 'viv.samsungCloudApp', 'viv.samsungHealthApp',
       'viv.samsungInternetApp', 'viv.samsungNotesApp', 'viv.samsungPayApp',
       'viv.samsungThemesApp', 'viv.scan3dApp', 'viv.smartSwitchApp',
       'viv.smartViewApp', 'viv.voiceRecorderApp', 'viv.worldClock']

capsule_remap_list = ['bixby.appstoreResolver', 'bixby.astrologyResolver_KR',
       'bixby.audioBookResolver_KR', 'bixby.bixbyChatResolver_KR',
       'bixby.briefing', 'bixby.callingResolver', 'bixby.carResolver_KR',
       'bixby.contentInfoResolver_KR', 'bixby.currencyResolver_KR',
       'bixby.deliveryResolver', 'bixby.deviceControlResolver_KR',
       'bixby.emailResolver_KR', 'bixby.flightResolver_KR', 'bixby.galaxyBuds',
       'bixby.howToUseBixby', 'bixby.issueReport', 'bixby.knowledgeResolver',
       'bixby.languageLearningResolver_KR', 'bixby.launcherResolver_KR',
       'bixby.mapResolver_KR', 'bixby.mediaResolver_KR',
       'bixby.messagingResolver', 'bixby.newsResolver_KR',
       'bixby.photoResolver_KR', 'bixby.podcastResolver_KR',
       'bixby.radioResolver_KR', 'bixby.recipeResolver',
       'bixby.restaurantResolver', 'bixby.routinesApp',
       'bixby.searchResolver_KR', 'bixby.socialMediaResolver',
       'bixby.sportsScoreResolver_KR', 'bixby.stockInfoResolver_KR',
       'bixby.systemResolver_KR', 'bixby.ticketsResolver',
       'bixby.translatorsResolver_KR', 'bixby.videoResolver_KR',
       'bixby.volumeResolver_KR', 'bixby.weatherInfoResolver_KR',
       'kbstar.liivon', 'samsung.announcement',
       'samsung.elderlyCareServiceApp', 'samsung.homeSecurity',
       'samsung.noActionAircon', 'samsung.speakerFindMyMobileApp',
       'viv.arEmojiApp', 'viv.arZoneApp', 'viv.bixbyChat_entertainment_koKR',
       'viv.bixbyHomeApp', 'viv.bixbyVisionApp', 'viv.bixbyVoiceApp',
       'viv.calculator', 'viv.calendarApp', 'viv.cameraApp', 'viv.clockApp',
       'viv.deviceFAQ', 'viv.deviceMaintenanceApp', 'viv.gracenote',
       'viv.interparkHotel', 'viv.lottery', 'viv.measurement',
       'viv.myFilesApp', 'viv.noActionAudioBook', 'viv.noActionDelivery',
       'viv.noActionEducation', 'viv.noActionFestival', 'viv.noActionFinance',
       'viv.noActionGame', 'viv.noActionGeneralSearch', 'viv.noActionMap',
       'viv.noActionMessage', 'viv.noActionMovie', 'viv.noActionNavi',
       'viv.noActionPlaceRecommendation', 'viv.noActionRadio',
       'viv.noActionRecipe', 'viv.noActionRelaxation',
       'viv.noActionShoppingRecommendation', 'viv.noActionSocialNetwork',
       'viv.noActionSpeakerCameraPhotos', 'viv.noActionSports',
       'viv.noActionTranslation', 'viv.noActionTransportation',
       'viv.noActionTv', 'viv.noActionVideo', 'viv.penupApp',
       'viv.reminderApp', 'viv.samsungAccountApp', 'viv.samsungCloudApp',
       'viv.samsungHealthApp', 'viv.samsungNotesApp', 'viv.samsungPayApp',
       'viv.samsungThemesApp', 'viv.scan3dApp', 'viv.smartSwitchApp',
       'viv.smartViewApp', 'viv.voiceRecorderApp', 'viv.worldClock']
capsule_remap_list_with_vivcore = ['bixby.appstoreResolver', 'bixby.astrologyResolver_KR',
       'bixby.audioBookResolver_KR', 'bixby.bixbyChatResolver_KR',
       'bixby.briefing', 'bixby.callingResolver', 'bixby.carResolver_KR',
       'bixby.contentInfoResolver_KR', 'bixby.currencyResolver_KR',
       'bixby.deliveryResolver', 'bixby.deviceControlResolver_KR',
       'bixby.emailResolver_KR', 'bixby.flightResolver_KR', 'bixby.galaxyBuds',
       'bixby.howToUseBixby', 'bixby.issueReport', 'bixby.knowledgeResolver',
       'bixby.languageLearningResolver_KR', 'bixby.launcherResolver_KR',
       'bixby.mapResolver_KR', 'bixby.mediaResolver_KR',
       'bixby.messagingResolver', 'bixby.newsResolver_KR',
       'bixby.photoResolver_KR', 'bixby.podcastResolver_KR',
       'bixby.radioResolver_KR', 'bixby.recipeResolver',
       'bixby.restaurantResolver', 'bixby.routinesApp',
       'bixby.searchResolver_KR', 'bixby.socialMediaResolver',
       'bixby.sportsScoreResolver_KR', 'bixby.stockInfoResolver_KR',
       'bixby.systemResolver_KR', 'bixby.ticketsResolver',
       'bixby.translatorsResolver_KR', 'bixby.videoResolver_KR',
       'bixby.volumeResolver_KR', 'bixby.weatherInfoResolver_KR',
       'kbstar.liivon', 'samsung.announcement',
       'samsung.elderlyCareServiceApp', 'samsung.homeSecurity',
       'samsung.noActionAircon', 'samsung.speakerFindMyMobileApp',
       'viv.arEmojiApp', 'viv.arZoneApp', 'viv.bixbyChat_entertainment_koKR',
       'viv.bixbyHomeApp', 'viv.bixbyVisionApp', 'viv.bixbyVoiceApp',
       'viv.calculator', 'viv.calendarApp', 'viv.cameraApp', 'viv.clockApp',
       'viv.deviceFAQ', 'viv.deviceMaintenanceApp', 'viv.gracenote',
       'viv.interparkHotel', 'viv.lottery', 'viv.measurement',
       'viv.myFilesApp', 'viv.noActionAudioBook', 'viv.noActionDelivery',
       'viv.noActionEducation', 'viv.noActionFestival', 'viv.noActionFinance',
       'viv.noActionGame', 'viv.noActionGeneralSearch', 'viv.noActionMap',
       'viv.noActionMessage', 'viv.noActionMovie', 'viv.noActionNavi',
       'viv.noActionPlaceRecommendation', 'viv.noActionRadio',
       'viv.noActionRecipe', 'viv.noActionRelaxation',
       'viv.noActionShoppingRecommendation', 'viv.noActionSocialNetwork',
       'viv.noActionSpeakerCameraPhotos', 'viv.noActionSports',
       'viv.noActionTranslation', 'viv.noActionTransportation',
       'viv.noActionTv', 'viv.noActionVideo', 'viv.penupApp',
       'viv.reminderApp', 'viv.samsungAccountApp', 'viv.samsungCloudApp',
       'viv.samsungHealthApp', 'viv.samsungNotesApp', 'viv.samsungPayApp',
       'viv.samsungThemesApp', 'viv.scan3dApp', 'viv.smartSwitchApp',
       'viv.smartViewApp', 'viv.voiceRecorderApp', 'viv.worldClock', 'viv.core']

## Named Dispatch 캡슐 분류 Dictionary 만들기
named_dispatch_capsule_remap_dic = {}
with open('/data/nebulach/workspace/bixby_training_data/named_dispatch_fool.txt','r') as myfile:
    num = 0
    pre_cap = ''
    postword = ['에서', '열고', '열어서']
    while True:
        line = myfile.readline()
        line = line.replace('\n','')
        if not line: break
        temp = line.split("\t")
        while '' in temp:
            temp.remove('')
        rep_cap = temp[0]
        if((num == 0) or (pre_cap != rep_cap)):
            named_dispatch_capsule_remap_dic[rep_cap] = {}
            for pw in postword:
                named_dispatch_capsule_remap_dic[rep_cap][temp[1].replace(' ','')+pw] = temp[2]
        elif(pre_cap == rep_cap):
            for pw in postword:
                named_dispatch_capsule_remap_dic[rep_cap][temp[1].replace(' ','')+pw] = temp[2]
#         print(rep_cap)
        pre_cap = rep_cap
        num = num+1


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/utt_sampling/', methods=['GET'])
def uttSampling():
    if request.method == 'GET':
        sample = user_utt.sample(10)

        sample_tokenized = [Tokenizers.PRETRAINED_256.value.tokenize(sentence) for sentence in sample['nltext_str']]
        # sample_result = BertClients.PRETRAINED_256_REMAP_CLS.value.encode(sample_tokenized, is_tokenized=True, show_tokens=False)
        sample_result_2 = BertClients.PRETRAINED_256_REMAP_PRECLS.value.encode(sample_tokenized, is_tokenized=True, show_tokens=False)

        test = []
        num_list = np.argsort(-sample_result_2)

        for i, utt in enumerate(sample['nltext_str']):
            num = i

            if(sample['capsule_remap'].tolist()[num] not in capsule_remap_list):
                test.append(UtteranceResult(sample['nltext_str'].tolist()[num], capsule_remap_list[num_list[num][0]],
                                            train_utt[
                                                train_utt['capsule_remap'] == capsule_remap_list[num_list[num][0]]].sample(3)[
                                                'nltext_str'].tolist(),
                                            capsule_remap_list[num_list[num][1]], train_utt[
                                                train_utt['capsule_remap'] == capsule_remap_list[num_list[num][1]]].sample(3)[
                                                'nltext_str'].tolist(),
                                            capsule_remap_list[num_list[num][2]], train_utt[
                                                train_utt['capsule_remap'] == capsule_remap_list[num_list[num][2]]].sample(3)[
                                                'nltext_str'].tolist(),
                                            capsule_remap_list[num_list[num][3]], train_utt[
                                                train_utt['capsule_remap'] == capsule_remap_list[num_list[num][3]]].sample(3)[
                                                'nltext_str'].tolist(),
                                            sample['capsule_remap'].tolist()[num],
                                            " ").toString())
            else:
                test.append(UtteranceResult(sample['nltext_str'].tolist()[num], capsule_remap_list[num_list[num][0]], train_utt[train_utt['capsule_remap'] == capsule_remap_list[num_list[num][0]]].sample(3)['nltext_str'].tolist(),
                capsule_remap_list[num_list[num][1]], train_utt[train_utt['capsule_remap'] == capsule_remap_list[num_list[num][1]]].sample(3)['nltext_str'].tolist(),
                capsule_remap_list[num_list[num][2]], train_utt[train_utt['capsule_remap'] == capsule_remap_list[num_list[num][2]]].sample(3)['nltext_str'].tolist(),
                capsule_remap_list[num_list[num][3]], train_utt[train_utt['capsule_remap'] == capsule_remap_list[num_list[num][3]]].sample(3)['nltext_str'].tolist(),
                sample['capsule_remap'].tolist()[num], train_utt[train_utt['capsule_remap'] == sample['capsule_remap'].tolist()[num]].sample(3)['nltext_str'].tolist()).toString())

        # test = []
        # for i, utt in sample.iterrows():
        #     test.append(UtteranceResult(utt['nltext_str'], utt['goal'], utt['goal'], utt['goal']).toString())

        return json.dumps([{"utts": test}])
        # return json.dumps([{"utt": sample['nltext_str'].tolist(), "goal": sample['goal'].tolist()}])

@app.route('/similar_utt_search_func/', methods=['POST'])
def similar_utt_search_func():
    if request.method == 'POST':
        query = request.form['search_utt']
        # topk = int(request.form['numTopUtts'])
        topk = 1000
        # useless_similarity_threshold = float(request.form['thresholdGM'])

        query_vec = BertClients.PRETRAINED_256_EMB.value.encode([Tokenizers.PRETRAINED_256.value.tokenize(query)], is_tokenized=True, show_tokens=False)
        print("query : ",query)
        score = cosine_similarity(query_vec.reshape(1,768), cluster_centers_decoded)
        cluster_idx = np.argsort(-score)[0][:2]
        if(cluster_idx[0] in [149,168,226,248]): cluster_idx = cluster_idx[1]
        else : cluster_idx = cluster_idx[0]

        print("cluster_idx : ",cluster_idx)
        print("cluster_num_list_size : ", len(cluster_num_list[cluster_idx]))

        cos_list = []
        with torch.no_grad():
            cluster_array = np.take(user_utt_encoded,np.array(cluster_num_list[cluster_idx]),axis=0)
            cos_list = cos(torch.from_numpy(cluster_array).float().to(device),torch.from_numpy(query_vec.reshape(1,768)).float().to(device))

        topk_idx_test = np.argsort(-cos_list)[:topk]
        cos_list = cos_list.cpu().numpy()
        results = []
        for i,idx in enumerate(topk_idx_test):
            results.append(
                SimilarUtteranceResult(
                    user_utt2['nltext_str'][cluster_num_list[cluster_idx][idx]],
                    user_utt2['count'][cluster_num_list[cluster_idx][idx]],
                    str(round(cos_list[idx], 4)), 
                    "202001", 
                    user_utt2['capsule'][cluster_num_list[cluster_idx][idx]],
                    user_utt2['goal'][cluster_num_list[cluster_idx][idx]],
                    # user_utt['capsule_remap'][cluster_num_list[cluster_idx][idx]],
                    user_utt2['device'][cluster_num_list[cluster_idx][idx]]
                ).toString())

    return json.dumps([{"similar_utts": results}])


@app.route('/beyond_evaluation_capsule_func/', methods=['GET'])
def beyond_evaluation_capsule_func():
    return json.dumps({"label": capsule_evaluation_list[4],"precision": capsule_evaluation_list[0],"recall": capsule_evaluation_list[1],"fscore": capsule_evaluation_list[2],"support": capsule_evaluation_list[3]})

@app.route('/beyond_mismatch_capsule_func/', methods=['GET'])
def beyond_mismatch_capsule_func():

    capsule_name = request.args.get('capsule_name')
    predicted_capsule_name = request.args.get('predicted_capsule_name')
    match_type = request.args.get('match_type')
    client_type = request.args.get('client_type')
    select_type = request.args.get('select_type')
    num_utt = int(request.args.get('num_utt'))
    print(predicted_capsule_name)

    search_query = ""
    if(capsule_name):
        search_query += ' & capsule_remap==\'' + capsule_name + '\''

    if(predicted_capsule_name):
        search_query += ' & predict==\'' + predicted_capsule_name + '\''
    if(match_type):
        search_query += ' & match==\'' + match_type + '\''
    if(client_type):
        search_query += ' & device==\'' + client_type + '\''
    if(search_query):
        search_query = search_query[2:]

    print(search_query)
    if(search_query):
        temp_user_utt = user_utt[user_utt.eval(search_query)]
    else:
        temp_user_utt = user_utt

    temp_user_utt_len = len(temp_user_utt)
    print(temp_user_utt_len)

    if(num_utt>temp_user_utt_len):
        num_utt = temp_user_utt_len

    if(select_type == 'random'):
        temp_user_utt = temp_user_utt.sample(num_utt)
    else:
        temp_user_utt = temp_user_utt.sort_values(by=['count'],ascending=False)[:num_utt]

    heatmap = []
    heatmap_num = []

    if(predicted_capsule_name):
        heatmap = confusion_result_log[capsule_remap_list_with_vivcore.index(predicted_capsule_name)].tolist()
        heatmap_num = confusion_result[capsule_remap_list_with_vivcore.index(predicted_capsule_name)].tolist()
    else:
        heatmap = ""
        heatmap_num = ""

    return json.dumps({"result_num":num_utt,"nltext_str": temp_user_utt['nltext_str'].tolist(), "goal": temp_user_utt['goal'].tolist(),
                       "device": temp_user_utt['device'].tolist(),"capsule": temp_user_utt['capsule'].tolist(),"capsule_remap": temp_user_utt['capsule_remap'].tolist(),
                       "duplication": temp_user_utt['count'].tolist(),
                       "predict": temp_user_utt['predict'].tolist(),"predict_1_p": temp_user_utt['predict_1_p'].tolist(),
                       "is_match": temp_user_utt['match'].tolist(), "predict_1": temp_user_utt['predict_1'].tolist(),"predict_2": temp_user_utt['predict_2'].tolist(),
                       "predict_2_p": temp_user_utt['predict_2_p'].tolist(),"predict_3": temp_user_utt['predict_3'].tolist(),"predict_3_p": temp_user_utt['predict_3_p'].tolist(),
                       "heatmap": heatmap,
                       "heatmap_num": heatmap_num,
                       "heatmap_name":predicted_capsule_name})

class StringConverter(dict):
    def __contains__(self, item):
        return True
    def __getitem__(self, item):
        return str
    def get(self, default=None):
        return str


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    for f in request.files.getlist('file'):
        f.save(os.path.join(APP_PATH+'/uploads/',f.filename))
        print(os.path.join(APP_PATH+'/uploads/',f.filename))
    #
    for f in request.files.getlist('file'):
        # bixby_ums_test = pd.read_csv(os.path.join(APP_PATH + '/uploads/', f.filename), delimiter=',',names=['num','nltext_str','capsule','converid','yyyymmdd'], converters=StringConverter(), low_memory=False )
        bixby_ums_test = pd.read_csv(os.path.join(APP_PATH + '/uploads/', f.filename), delimiter=',',names=['num','nltext_str','capsule','converid','yyyymmdd'], low_memory=False )
        bixby_ums_test['nltext_str'] = bixby_ums_test['nltext_str'].astype(str)
        bixby_ums_test = predict_utts(bixby_ums_test)
        # bixby_ums_test['strong_true'] = (bixby_ums_test['predict_1_p'] > 12)
        # bixby_ums_test['more_loose_false'] = (bixby_ums_test['predict_1_p']<10)
        # bixby_ums_test['loose_false'] = (bixby_ums_test['predict_1_p']<10) & ~((bixby_ums_test['predict_1_p']>7)&(bixby_ums_test['predict_2_p']>6))
        # bixby_ums_test['strong_false'] = (bixby_ums_test['predict_1_p']<10) & ~((bixby_ums_test['predict_1_p']>6)&(bixby_ums_test['predict_2_p']>5))

        # bixby_ums_test['strong_true'] = (bixby_ums_test['1predict_1_p'] >12 ) & (bixby_ums_test['1predict_1'] == bixby_ums_test['predict_1']) & (bixby_ums_test['1predict_1'] == bixby_ums_test['2predict_1'])& (bixby_ums_test['2predict_1'] == bixby_ums_test['3predict_1'])
        # bixby_ums_test['loose_true'] = (bixby_ums_test['1predict_1_p'] >8 ) & (bixby_ums_test['1predict_1'] == bixby_ums_test['predict_1']) & (bixby_ums_test['1predict_1'] == bixby_ums_test['2predict_1'])& (bixby_ums_test['2predict_1'] == bixby_ums_test['3predict_1'])
        # bixby_ums_test['more_loose_true'] = (bixby_ums_test['predict_1_p'] > 8)
        # bixby_ums_test['more_loose_false'] = (bixby_ums_test['1predict_1_p']<10)
        # bixby_ums_test['loose_false'] = (bixby_ums_test['1predict_1_p']<10) & ~((bixby_ums_test['1predict_1_p']>7)&(bixby_ums_test['1predict_2_p']>6))
        # bixby_ums_test['strong_false'] = (bixby_ums_test['1predict_1_p']<10) & ~((bixby_ums_test['1predict_1_p']>6)&(bixby_ums_test['1predict_2_p']>5))
        # bixby_ums_test = bixby_ums_test.drop(columns=['num','predict_1_p','predict_2','predict_2_p','predict_3','predict_3_p'])
        bixby_ums_test.to_csv(os.path.join(APP_PATH + '/uploads/', 'result_'+f.filename),encoding='utf-8',sep=',',index=False)

    return json.dumps({"success":True})


def predict_utts(utt_df):
    token_list = [Tokenizers.PRETRAINED_256.value.tokenize(sentence) for sentence in utt_df['nltext_str']]
    
    test_outlier_remap_pre = BertClients.PRETRAINED_256_REMAP_PRECLS_NEW.value.encode(token_list, is_tokenized=True, show_tokens=False)
    num_list = np.argsort(-test_outlier_remap_pre)
    num_list = num_list[:,0:3]

    utt_df['predict_1'] = [capsule_remap_list_new[i] for i in num_list[:,0]]
    utt_df['predict_1_p'] = [test_outlier_remap_pre[i][x] for i,x in enumerate(num_list[:,0])]
    utt_df['predict_2'] = [capsule_remap_list_new[i] for i in num_list[:,1]]
    utt_df['predict_2_p'] = [test_outlier_remap_pre[i][x] for i,x in enumerate(num_list[:,1])]
    utt_df['predict_3'] = [capsule_remap_list_new[i] for i in num_list[:,2]]
    utt_df['predict_3_p'] = [test_outlier_remap_pre[i][x] for i,x in enumerate(num_list[:,2])]

    test_outlier_remap_pre = BertClients.PRETRAINED_256_REMAP_PRECLS_NEW2.value.encode(token_list, is_tokenized=True, show_tokens=False)
    num_list = np.argsort(-test_outlier_remap_pre)
    num_list = num_list[:,0:3]
    utt_df['prediction'] = [capsule_remap_list_new[i] for i in num_list[:,0]]
    # utt_df['1predict_1_p'] = [test_outlier_remap_pre[i][x] for i,x in enumerate(num_list[:,0])]
    # utt_df['1predict_2'] = [capsule_remap_list[i] for i in num_list[:,1]]
    # utt_df['1predict_2_p'] = [test_outlier_remap_pre[i][x] for i,x in enumerate(num_list[:,1])]
    # utt_df['1predict_3'] = [capsule_remap_list[i] for i in num_list[:,2]]
    # utt_df['1predict_3_p'] = [test_outlier_remap_pre[i][x] for i,x in enumerate(num_list[:,2])]

    for i in utt_df.index:
        if(utt_df.loc[i,'prediction'] in named_dispatch_capsule_remap_dic.keys()):
            for key in named_dispatch_capsule_remap_dic[utt_df.loc[i,'prediction']].keys():
                if(utt_df.loc[i,'nltext_str'].replace(' ','')[:len(key)] == key):
                    utt_df.loc[i,'prediction'] = named_dispatch_capsule_remap_dic[utt_df.loc[i,'prediction']][key]

    utt_df['result'] = 4
    utt_df.loc[(utt_df['predict_1_p']<10), 'result'] = 3
    utt_df.loc[(utt_df['predict_1_p']<10) & ~((utt_df['predict_1_p']>7)&(utt_df['predict_2_p']>6)), 'result'] = 2
    utt_df.loc[(utt_df['predict_1_p']<10) & ~((utt_df['predict_1_p']>6)&(utt_df['predict_2_p']>5)), 'result'] = 1
    utt_df.loc[(utt_df['predict_1_p'] > 12) & (utt_df['predict_1'] == utt_df['prediction']), 'result'] = 5
    utt_df = utt_df.drop(
        columns=['num', 'predict_1', 'predict_1_p', 'predict_2', 'predict_2_p', 'predict_3', 'predict_3_p'])

    return utt_df

@app.route('/result_file_download', methods=['GET'])
def result_file_downloada():
    file_name_ori = request.args.get('file_name')
    aa = os.path.join(APP_PATH + '/uploads/','result_'+file_name_ori)
    file_name_ori = 'result_' + file_name_ori
    print("aaa", file_name_ori)
    print("bbb",aa)

    return send_file(aa,
                     mimetype='text/csv',
                     attachment_filename=file_name_ori,
                     as_attachment=True);

@app.route('/result')
def result():
    return render_template("result.html")
@app.route('/result_bar_256_top_100/')
def result_bar_256_top_100():
    return render_template('result_bar_256_top_100.html')
@app.route('/result_bar_256_top_3/')
def result_bar_256_top_3():
    return render_template('result_bar_256_top_3.html')
@app.route('/result_scatter_256_top_100/')
def result_scatter_256_top_100():
    return render_template('result_scatter_256_top_100.html')
@app.route('/result_scatter_256_top_3/')
def result_scatter_256_top_3():
    return render_template('result_scatter_256_top_3.html')
@app.route('/utterance_plotting/')
def utterance_quality():
    return render_template('utterance_plotting.html')
@app.route('/utterance_ave_distance/')
def utterance_ave_distance():
    return render_template('utterance_ave_distance.html')
@app.route('/capsule_mismatch_inspection/')
def capsule_mismatch_inspection():
    return render_template('capsule_mismatch_inspection.html')
@app.route('/beyond_evaluation_capsule/')
def beyond_evaluation_capsule():
    return render_template('beyond_evaluation_capsule.html')
@app.route('/beyond_mismatch_capsule/')
def beyond_mismatch_capsule():
    return render_template('beyond_mismatch_capsule.html')
@app.route('/similar_utt_search/')
def similar_utt_search():
    return render_template('similar_utt_search.html')
@app.route('/beyond_for_newcc/')
def beyond_for_newcc():
    return render_template('beyond_for_newcc.html')

if __name__ == '__main__':
    app.run(host="caffe.nebulach.com",debug=False, use_reloader=False, port=10204)
