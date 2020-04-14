import os
import cv2
import pickle
import numpy as np
from nltk.translate.meteor_score import single_meteor_score
import sys
sys.path.append("../")
from metrics import bleu, jiwer, rouge

def __load_frame__(frame_path, size = None, channels = 3):
        if channels == 1:
            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        elif channels == 3:
            img =  cv2.cvtColor(cv2.imread(frame_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)

        if size:
            return cv2.resize(img, tuple(size))
        else:
            return img

def save_predictions(model, path_to_save, vocab, args, train_data, test_data, dev_data = None):
    """train_data, test_data and dev_data must be a tensorflow Dataset returning (video, 
        sentence, path_video) respectively. The tensorflow dataset must be batched with 
        and only with one example.
    """

    results = {}

    index_word = vocab.index_word

    types_data = [("train", train_data), ("test", test_data)]
    if dev_data:
        types_data.append(("dev", dev_data))

    for type_data, gen in types_data:
        results[type_data] = {}
        for video, sentence, path in gen:
            # Check for batch size to be 1
            if video.shape[0] != 1 or sentence.shape[0] != 1 or path.shape[0] != 1:
                raise AssertionError('The data in the generator is bad batched...')

            # Convert the array to word
            target_sentence = [index_word[i] for i in sentence[0].numpy()][1:]
            print("Video: " + path[0].numpy().decode('utf-8'))
            print("Reference: " + " ".join(target_sentence))

            # Predict in the model
            prediction_indexes = model.predict((video, sentence[:,:-1]))
            prediction_indexes = np.argmax(prediction_indexes[0], axis=1)
            prediction_sentence = [index_word[i] for i in prediction_indexes]
            
            print("Translation: "+" ".join(prediction_sentence)+"\n")

            results[type_data][path[0].numpy().decode('utf-8')] = {"prediction_sentence" : prediction_sentence,
                "target_sentence" : target_sentence
                }
    
    with open(os.path.join(path_to_save, "results.pkl"),'wb') as file:
        pickle.dump(results, file)

    return results

def calculate_metrics_results(results : dict):
    for data in results.keys():
        references = []
        translations = []
        references_rouge = []
        translations_rouge = []
        wert = 0.0
        meteort = 0.0
        for video in results[data].keys():
            translation = results[data][video]["prediction_sentence"]
            if '</s>' in translation:
                translation.remove('</s>')
            translation = " ".join(translation)

            reference = results[data][video]["target_sentence"]
            if '</s>' in reference:
                reference.remove('</s>')
            reference = " ".join(reference)

            wert += jiwer.wer(truth = reference, hypothesis = translation) 
            meteort += single_meteor_score(reference, translation)

            translations.append(translation.split(" "))
            translations_rouge.append(translation)

            references.append([reference.split(" ")])
            references_rouge.append(reference)

        print(len(references))
        rouge_score_map = rouge.rouge(translations_rouge, references_rouge)
        print(data + ' rouge: ' + str(100 * rouge_score_map["rouge_l/f_score"]))
        print(data + ' WER: ' + str((wert/len(references))*100))
        print(data + ' Meteor: ' + str((meteort/len(references))*100))
        for max_ in range(1,5):
            bleu_score, _, _, _, _, _ = bleu.compute_bleu(references, translations, max_order=max_)
            print(data + ' bleu: ' + str(max_) + " " + str(bleu_score*100))