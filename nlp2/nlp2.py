#!/usr/bin/env python
# coding: utf-8

# #                               2η Εργαστηριακή Άσκηση 
# ### Ονοματεπώνυμο : Σιφναίος Σάββας
# ### ΑΜ : 031 16080

# # 2. Θεωρητικό Υπόβραθρο 
# 
# Κατά την εκπαίδευση ενός συστήματος αναγνώρισης φώνης, κρίνεται απαραίτητη η εξαγωγή ορισμένων χαρακτηριστικών από το σήμα φωνής. Στο συγκεκριμένο εργαστήριο, για την απεικόνιση των χαρακτηριστικών αυτών θα χρησιμοποιήσουμε τα MFCC Vectors. Η παραπάνω ιδέα των MFCC βασίζεται αφενός σε αρχές και έννοιες της θεωρίας του cepstrum και αφετέρου σε αποτελέσματα ψυχοακουστικών μελετών που πραγματοποιήθηκαν για τον εντοπισμό των συχνοτικών εκεινών ζωνών που γίνονται "ευκολότερα"  αντιληπτές από τον ανθρώπινο αυτί. Σκόπος των μελετών αυτών, είναι βέλτιστη κωδικοποιήση των ηχητικών σημάτων.
# 
# Η εξαγωγή των χαρακτηριστικών των ηχητικών σημάτων εισόδου, μπορεί να οργανωθεί στα εξής βήματα : 
# 
# 1) Προέμφαση του ήχου μέσα από την μετατόπιση κάθε φωνήματος σε υψηλότερες συχνότητες που βοηθούν την αναγνώριση.
# 
# 2) Παραθυροποίηση των ηχητικών δεδομένων για να διατηρήσουμε την στατικότητα των στατιστικών χαρακτηριστικών του ήχου(μέση τιμή, διακύμανση) και κατ’ επέκταση την εργοδικότητα (αφού έχουμε πλήθος δεδομένων στο χρόνο αλλά δε μπορούμε να επαναλάβουμε διαδοχικά τις ίδιες εκτελέσεις των ίδιων δεδομένων).Συχνά χρησιμοποιούμε Kaiser ή Hamming παράθυρα ,διάρκειας από 15ms εως 30ms, με δεδομένο overlap (περίπου 40-60%) για να ομαλοποιηθούν φαινόμενα των άκρων.
# 
# 3) Συχνοτική ανάλυση σήματος ,μέσω της χρήσης DFT, για τον υπολογισμό της συγκέντρωσης της ενέργειας σε κάθε συχνότητα.
# 
# 4) Mel filter bank διαμόρφωση,που αιτιολογείται επειδή ο ανθρώπινος εγκέφαλος παρουσιάζει δυσκολία στον διαχωρισμό συχνοτήτων σε υψηλοσυχνοτικά σήματα.Γενικά η ανθρώπινη απόκριση είναι γραμμική κάτω από τα 1000Hz και λογαριθμική παραπάνω,οπότε έτσι σχεδιάονται και τα φίλτρα μας.
# 
# 5) Χρήση cepstrum, για την μετάθεση του σήματος σε πεδίο quefrency και το διαχωρισμό πηγής και φίλτρου. Αποδεικνύεται ότι ο διαχωρισμός αυτός βελτιώνει το μοντέλο καθώς ο εντοπισμός π.χ. του φίλτρου: Θέση της γλώσσας μπορεί να προσφέρει στην ανάλυση παραπάνω χαρακτηριστικά για την αναγνώριση των φωνημάτων.
# 
# 6) Delta-Training: Τα χαρακτηριστικά αυτά είναι τιμές που δείχνουν πόσο μεταβάλλονται κάποια στατιστικά χαρακτηριστικά του cepstrum μέσα σε ένα παράθυρο. Συμπληρώνουν το διάνυσμα που δημιουργούμε.Για ακρίβεια χρησιμοποιούμε και τα double-deltas που δείχνουν την μεταβολή των deltas . Αυτός ο χειρισμός του σήματος καταλήγει σε MFC Coeficients και περιέχει το μεγαλύτερο μέρος της πληροφορίας για κάθε παραθυροποιημένο σήμα.
# 
# • Γλωσσικό Μοντέλο (Language Model LM). Για το μοντέλο αυτό συνήθως χρησιμοποιούμε unigrams και bigrams. Ωστόσο, κατά την χρησιμοποίηση του Kaldi για την δημιουργία του μοντέλου αναγνώρισης φωνής βλέπουμε ότι δημιουργεί unigram, bigram συνδυάζοντάς τα και προφανώς αυξάνοντας την υπολογιστική πολυπλοκότητα αλλά και την ακρίβεια όσο αυξάνεται το μέγεθος του μοντέλου.
# 
# • Ακουστικό Μοντέλο (Acoustic Model AM). Στο στάδιο της δημιουργίας του ακουστικού μοντέλου ουσιαστικά υπολογίζουμε την πιο πιθανή ακολουθία από παρατηρήσεις όταν μας έχουν δοθεί άλλα γλωσσικά χαρακτηριστικά όπως (λέξεις, φωνήματα ή κάποια μέρη φωνημάτων). Πιο συγκεκριμένα δοθέντος μίας πιθανότητας P(O|W) μπορούμε να κατατάξουμε για κάθε HMM μίας κατάστασης q του αυτομάτου μας, χρησιμοποιώντας Gaussian Mixture Models (GMM’s) και τα υπόλοιπα γλωσσικά χαρακτηριστικά, την πιθανοφάνεια.

# # 3.    Βήματα Προπαρασκεύης 
# 
# Αρχικά, στο φάκελο egs, δημιουργούμε ένα νέο φάκελο με όνομα 'usc'. Στον φάκελο 'usc', δημιουργούμε τον φάκελο 'data'.Στην συνέχεια, μέσα σε αυτό το φάκελο δημιουργούμε τους υποφακέλους 'train', 'test' , 'dev' στους οποίους θα αποθηκέυσουμε στην συνέχεια τα αντίστοιχα αρχεία 'uttids' , 'utt2spk' , 'wav.scp' , 'text' ,που θα χρησιμοποιηθούν για την κατασκευή του γλωσσικού και του ακουστικού μοντέλου.

# In[244]:


get_ipython().run_cell_magic('bash', '', '\nmkdir /users/savas/kaldi/egs/usc\ncd /users/savas/kaldi/egs/usc\nmkdir data\nmkdir data/train \nmkdir data/test\nmkdir data/dev')


# In[176]:


import re

global_path = "/Users/Savas/kaldi/egs/usc/data/"
train_path = global_path +"train"
test_path = global_path +"test"
dev_path = global_path +"dev"
paths = {'train':train_path,'test':test_path,'dev':dev_path}


# Πριν προχωρήσουμε στην δημιουργία των παραπάνω αρχείων, πραγματοποιούμε μια μορφή προεπεξεργάσιας των δεδομένων μας, τα οποια βρίσκονται αποθηκευμένα στον φάκελο 'slp_lab2_data'. Για τα αρχεία 'train_utterances.txt' , 'test_utterances.txt' , 'validation_utterances.txt' , προβαίνουμε σε αντικατάσταση των συμβόλων "_" με το      " " (κενό) και το αποτέλεσμα της αντικατάστασης αυτής το αποθηκεύουμε στα αρχεία 'train_utt.txt', 'test_utt.txt', 'val_utt.txt' , αντίστοιχα. Τα αρχεία αυτά αποτέλουν μια προσωρινή , ευκολότερα επεξεργάσιμη μορφή των αρχείων που τα δημιούργησαν. Στην σύνεχεια, από την ενδιάμεση αυτή μορφή απομονώνουμε το τρίτο πεδίο και το αποθηκεύουμε στα αρχεία 'train_utter.txt' , 'test_utter.txt' , 'val_utter.txt'. Τα τελευταία τρία αρχεία, έχουν αποθηκευμένα σε κάθε γραμμή τον ομιλητή που εκφωνεί την συγκεκριμένη πρόταση στο αντίστοιχο set αρχείων. Επιπλέον, στα αρχεία 'sample_train_utter.txt' , 'sample_test_utter.txt' , 'sample_val_utter.txt', αποθηκεύουμε το τέταρτο πεδίο από τα αρχεία 'train_utt.txt', 'test_utt.txt', 'val_utt.txt' αντίστοιχα. Στα τελευταία αυτά αρχεία, περιέχεται ,ανά γραμμή, ο αριθμός κάθε πρότασης για κάθε ένα από τα τρία σύνολα δεδομένων. Τέλος, επεξεργαζόμαστε κατάλληλα και το αρχείο 'lexicon.txt'. Συγκεκριμένα, απομονώνουμε το πρώτο και το δεύτερο πεδίο του αρχείου αυτού αποθηκεύοντας τα πεδία αυτά στα αρχεία 'indices_lexicon.txt' , 'temp.txt' αντίστοιχα. Το αρχείο 'temp.txt' θα ήταν κατάλληλο για χρήση στην συνέχεια, ωστόσο στην αρχή κάθε γραμμής περιέχει ένα χαρακτήρα κενού, ο οποίος ενδεχομένως να προκαλέσει προβλήματα στην συνέχεια. Για τον λόγο αυτό, με την εντολή 'cut -b 2- ,απορρίπτουμε το πρώτο byte κάθε γραμμής (που αντιστοιχεί στον χαρακτήρα του κενού) και το αποτέλεσμα το αποθηκεύουμε στο αρχείο 'values_lexicon.txt'.

# In[177]:


get_ipython().run_cell_magic('bash', '', '\n#Replace "_" with spaces and save the output to \'*_utt.txt\' files \ncat ./slp_lab2_data/filesets/train_utterances.txt |tr "_" [:space:] > ./slp_lab2_data/filesets/train_utt.txt \ncat ./slp_lab2_data/filesets/test_utterances.txt |tr "_" [:space:] > ./slp_lab2_data/filesets/test_utt.txt\ncat ./slp_lab2_data/filesets/validation_utterances.txt |tr "_" [:space:] > ./slp_lab2_data/filesets/val_utt.txt\n\n#From \'lexicon.txt\' we save the first and second field in seperate files\ncut -f 1 ./slp_lab2_data/lexicon.txt > ./slp_lab2_data/indices_lexicon.txt\ncut -f 2 ./slp_lab2_data/lexicon.txt > ./slp_lab2_data/temp.txt\n#From the file containing the second field of \'lexicon.txt\' we save every byte from the second one till the last one \ncut -b 2- ./slp_lab2_data/temp.txt > ./slp_lab2_data/values_lexicon.txt\n\n#Create new \'*_utter.txt\' files containing only the 3rd field of \'*_utt.txt\' files accordingly\ncut -f 3 ./slp_lab2_data/filesets/train_utt.txt > ./slp_lab2_data/filesets/train_utter.txt \ncut -f 3 ./slp_lab2_data/filesets/test_utt.txt > ./slp_lab2_data/filesets/test_utter.txt \ncut -f 3 ./slp_lab2_data/filesets/val_utt.txt > ./slp_lab2_data/filesets/val_utter.txt \n\n#Create new \'*_utter.txt\' files containing only the 4th field of \'*_utt.txt\' files accordingly\ncut -f 4 ./slp_lab2_data/filesets/train_utt.txt > ./slp_lab2_data/filesets/sample_train_utter.txt \ncut -f 4 ./slp_lab2_data/filesets/test_utt.txt > ./slp_lab2_data/filesets/sample_test_utter.txt \ncut -f 4 ./slp_lab2_data/filesets/val_utt.txt > ./slp_lab2_data/filesets/sample_val_utter.txt ')


# Για την κατασκευή των αρχείων 'uttids' , 'utt2spk' , 'wav.scp' , 'text', ορίζουμε την συνάρτηση 'create_files' ,η οποία δέχεται σαν ορίσμα ενα path προς το αντίστοιχο directory στο οποίο θα αποθηκευθούν τα αρχεία αυτά. Παρακάτω, ακολουθεί σύντομη περιγραφή του τρόπου δημιουργίας κάθε ενός από τα παραπάνω αρχεία.
# 
# -uttids  : με την εντολή 'open(path+"/uttids","w+")', δημιουργούμε στον φάκελο που υπαγορεύεται από την μεταβλητή 'path' ένα κενό αρχείο με όνομα 'uttids'. Συνεχίζοντας, για κάθε ένα από τα τρία σύνολα ανοίγουμε , με δικαιώματα ανάγνωσης, το κατάλληλο αρχείο *_utterances.txt , όπου * = {train, test, dev} , το διαβάζουμε και το αποθηκεύουμε σε μια λίστα και τέλος, "γράφουμε την λίστα αυτή στο αρχείο uttids. Με αυτό τον τρόπο, επιτυγχάνουμε την αντιγραφή του περιεχόμενου του αρχείου *_utterances.txt στο αρχείο uttids, για κάθε ένα από τα τρία συνόλο δεδομένων. 
# 
# -utt2spk : Με την εντολή 'open(path+"/utt2spk","w+")' , δημιουργούμε στον φάκελο που υπαγορεύεται από την μεταβλητή 'path' ένα κενό αρχείο με όνομα 'utt2spk'. Συνεχίζοντας, διαβάζουμε το αρχείο *_utter.txt , που δημιρουργήσαμε παραπάνω,αποθηκεύοντας κάθε γραμμή του αρχείου ως στοιχείο της λίστας glines. Επιπλέον, χρησιμοποιούμε ένα μετρητή , οι τιμές του οποίου κυμένονται από το 1 εώς και το πλήθος των σειρών του κάθε αρχείου. Τέλος, για κάθε στοιχείο της λίστας γράφουμε στο αρχείο utt2spk "utternace_id_cnt glines[i]" , όπου cnt η τιμή του μετρητή (που αντιστοιχεί στην γραμμή που επεξεργαζόμαστε) και glines[i] το περιεχόμενο της συγκεκριμένης γραμμής.
# 
# -wav.scp : Με την εντολή 'open(path+"/utt2spk","w+")' , δημιουργούμε στον φάκελο που υπαγορεύεται από την μεταβλητή 'path' ένα κενό αρχείο με όνομα 'wav.scp'. Ακολουθούμε στην συνέχεια την ίδια διαδικασία με παραπάνω , μόνο που αυτή την φορά στο αρχείο 'wav.scp' αντί για glines[i] γράφουμε το path προς το αντίστοιχο αρχείου ήχου. 
# 
# -text : Με την εντολή 'open(path+"/utt2spk","w+")' , δημιουργούμε στον φάκελο που υπαγορεύεται από την μεταβλητή 'path' ένα κενό αρχείο με όνομα 'text'. Η συμπλήρωση του αρχείου text απαιτεί αρκετές ενέργειες παραπάνω. Αρχικά, διαβάζουμε κάθε γραμμή του αρχείου transcription.txt και την αποθηκευούμε στην λίστα dlines. Επαναλαμβάνουμε την ίδια διαδικάσια για τα αρχεία indices_lexicon.txt και values_lexicon.txt, αποθηκεύοντας τις γραμμές των αρχείων αυτών στις λίστες keys και values αντίστοιχα. Κάθε στοιχείο της λίστας dlines αποτελεί και μια από τις προτάσεις που λένε οι 4 ομιλητές. Στην συνέχεια, για κάθε στοιχείο της dlines αντικαθιστούμε τους κεφαλαίους χαρακτήρες με τους αντίστοιχους "μικρόυς" τους. Επιπλέον, αντικαθηστούμε κάθε ειδικό χαρακτήρα εκτός του " ' " με τον χαρακτήρα του κενού και τέλος χωρίζουμε κάθε πρόταση σε λέξεις, αποθηκευόντας το αποτέλεσμα ξανά στο αντίστοιχο στοιχείο της dlines. Μετα την ολοκλήρωση της παραπάνω διαδικασίας, η dlines είναι μια λίστα που περιέχει λίστες και η κάθε "υπολίστα" περιέχει τα tokens κάθε πρότασης. Στην συνέχεια, ορίζουμε ενα dictionary "lexicon" τα keys του οποίου είναι το περιεχόμενο της λίστας keys και τα values του το αντίστοιχο στοιχείο της λίστας values. Το dictionary αυτό, πρόκειται ουσιαστικά για μία αντιστοιχία κάθε λέξης σε μια ακολουθία φωνημάτων που "περιγράφουν" την λέξη αυτή. Στην συνέχεια, με την βοήθεια του dictionary , που δημιουργήσαμε, και της τροποποιημένης dlines ορίζουμε την λίστα pho , της οποίας κάθε στοιχείο είναι η ακολουθία των φωνημάτων που "περιγράφουν" την πρόταση που είναι αποθηκεύμενη στην λίστα dlines. Επιπλέον, στην αρχή και στο τέλος κάθε string της λίστας pho προσθέτουμε το φώνημα sil που αντιστοιχεί στην σιωπή. Τέλος, γράφουμε στο αρχείο text "utterance_id_cnt pho[i]" , όπου cnt είναι ο ίδιος με παραπάνω μετρητής και pho[i] το i-οστο στοιχείο της λίστας pho.   

# In[187]:


def create_files(path) :
    
    f1 = open(path+"/uttids","w+")
    f2 = open(path+"/utt2spk","w+")
    f3 = open(path+"/wav.scp","w+")
    f4 = open(path+"/text","w+")
    f5 = open("./slp_lab2_data/indices_lexicon.txt","r")
    f6 = open("./slp_lab2_data/values_lexicon.txt","r")
    d = open("./slp_lab2_data/transcription.txt","r")
    
    dlines = d.readlines()
    keys = f5.readlines()
    values = f6.readlines()
    
    d.close()
    f5.close()
    f6.close()
    
    for i in range(0,len(dlines)) :
        dlines[i] = dlines[i].lower()
        dlines[i] = re.sub('[^a-zA-Z0-9\'\s]',' ',dlines[i])
        dlines[i] = dlines[i].strip() # not sure if needed 
        dlines[i]=dlines[i].split()
    
    for i in range(0,len(keys)):
        keys[i] = keys[i].lower()
        keys[i] = keys[i].strip('\n')
        
    lexicon = {}
    for i in range (0,len(keys)):
        lexicon[keys[i]] = values[i]
    
    pho = ["sil "]*len(dlines)
    for i in range(0,len(dlines)):
        for word in dlines[i]:
            pho[i] += lexicon[word]
            
    for i in range(0,len(pho)):
        pho[i] = pho[i].replace('\n',' ')
        pho[i] += 'sil' 
        
    if path == train_path:
        f = open("./slp_lab2_data/filesets/train_utterances.txt","r")
        g = open("./slp_lab2_data/filesets/train_utter.txt","r")
        n = open("./slp_lab2_data/filesets/sample_train_utter.txt","r")
        
        flines = f.readlines()
        glines = g.readlines()
        nlines = n.readlines()
           
        f.close()
        g.close()
        n.close()
        d.close()
        
        cnt= 1
        for i in range(0,len(flines)) :
            f1.write(flines[i])
            f2.write("utterance_id_"+format(cnt,'04d')+" "+glines[i])
            f3.write("utterance_id_"+format(cnt,'04d')+" "+"/Users/Savas/downloads/slp_lab2_data/wav/"+glines[i][0:len(glines[i])-1]+"/"+flines[i][0:len(flines[i])-1]+".wav"+"\n")
            f4.write("utterance_id_"+format(cnt,'04d')+" "+pho[int(nlines[i][0:len(nlines[i])-1])-1]+'\n')
            cnt+=1
            
    elif path == test_path :
        f = open("./slp_lab2_data/filesets/test_utterances.txt","r")
        g = open("./slp_lab2_data/filesets/test_utter.txt","r")
        n = open("./slp_lab2_data/filesets/sample_test_utter.txt","r")
        
        flines = f.readlines()
        glines = g.readlines()
        nlines = n.readlines()
        
        f.close()
        g.close()
        n.close()
        
        cnt=1
        for i in range(0,len(flines)) :
            f1.write(flines[i])
            f2.write("utterance_id_"+format(cnt,'03d')+" "+glines[i])
            f3.write("utterance_id_"+format(cnt,'03d')+" "+"/Users/Savas/downloads/slp_lab2_data/wav/"+glines[i][0:len(glines[i])-1]+"/"+flines[i][0:len(flines[i])-1]+".wav"+"\n")
            f4.write("utterance_id_"+format(cnt,'03d')+" "+pho[int(nlines[i][0:len(nlines[i])-1])-1]+'\n')
            cnt+=1   
            
    elif path == dev_path :
        f = open("./slp_lab2_data/filesets/validation_utterances.txt","r")
        g = open("./slp_lab2_data/filesets/val_utter.txt","r")
        n = open("./slp_lab2_data/filesets/sample_val_utter.txt","r")
        
        flines = f.readlines()
        glines = g.readlines()
        nlines = n.readlines()
        
        f.close()
        g.close()
        n.close()
        
        cnt=1
        for i in range(0,len(flines)) :
            f1.write(flines[i])
            f2.write("utterance_id_"+format(cnt,'03d')+" "+glines[i])
            f3.write("utterance_id_"+format(cnt,'03d')+" "+"/Users/Savas/downloads/slp_lab2_data/wav/"+glines[i][0:len(glines[i])-1]+"/"+flines[i][0:len(flines[i])-1]+".wav"+"\n")
            f4.write("utterance_id_"+format(cnt,'03d')+" "+pho[int(nlines[i][0:len(nlines[i])-1])-1]+'\n')
            cnt+=1
    
    f1.close()
    f2.close()
    f3.close()
    f4.close()


# In[188]:


for i in paths :
    create_files(paths[i])


# # 4. Βήματα Κυρίως Μέρους
# 

# ## 4.1 Προετοιμασία διαδικασίας αναγνώρισης φωνής για τη USC-TIMIT
# 
# 1. Με τις εντολές "cp -p /users/savas/kaldi/egs/wsj/s5/path.sh /users/savas/kaldi/egs/usc" και "cp -p /users/savas/kaldi/egs/wsj/s5/cmd.sh /users/savas/kaldi/egs/usc" αντιγράφουμε τα αρχεία "path.sh" και "cmd.sh" αντίστοιχα, από τον φάκελο wsj/s5 στον φάκελο εργασίας μας egs/usc. Στην συνέχεια , χωρίς την χρήση εντολών bash scripting, τροποποιούμε την τιμή της μεταβλητής KALDI_ROOT , στο path.sh ώστε να ταυτίζεται με το path προς το κέντρικό φάκελο που είναι αποθηκεύμενο το Kaldi. Επιπλέον, αντικαθηστούμε την τιμή των μεταβλητών train_cmd , decode_cmd , cuda_cmd που βρίσκονται στο cmd.sh από queue.pl σε run.pl
# 
# 2. Με την εντολή "ln -s source destination" δημιουργούμε soft links εντός του φακέλου usc,όπου και εργαζόμαστε, που δείχνουν τόσο προς τον φάκελο steps που βρίσκεται στο directory kaldi/egs/wsj/s5/steps όσο και προς τον φάκελο utils που βρίσκεται στο directory kaldi/egs/wsj/s5/utils
# 
# 3. Επιπλέον, κάνουμε χρήση της εντολής "mkdir /users/savas/kaldi/egs/usc/local" για την δημιουργία του φακέλου local ,εντός του usc, μέσα στον οποίο με την εντολή "ln -s source destination" δημιουργούμε ένα ακόμα soft link το οποίο δείχνει αυτή την φορά προς το directory kaldi/egs/wsj/s5/steps/score_kaldi.sh . Το αρχείο score_kaldi.sh θα χρησιμοποιηθεί στην συνέχεια για τον υπολογισμό του PER (Phone Error Rate) 
# 
# 4. Στην συνέχεια με την "mkdir /users/savas/kaldi/egs/usc/conf" δημιουργούμε τον υπο-φάκελο local μέσα στον οποίο αντιγράφουμε το αρχείο mfcc.conf ,με την εντολή "cp -p /users/savas/downloads/lab2_help_scripts/mfcc.conf /users/savas/kaldi/egs/usc/conf", το οποίο δίνεται στις διευκρινίσεις. Tο αρχείο αυτό θα χρησιμοποιηθεί για τον υπολογισμό των MFCC features, με συχνότητα δειγματοληψίας f = 22050Hz. Γενικά η εντολή steps/makefcc.sh, αναζητά για το αρχείο conf/mfcc.conf στον ”conf” για οποιαδήποτε μη default παράμετρο, οι οποίες προωθούνται ως ορίσματα στα αντίστοιχα binary αρχεία.
# 
# 5. Τέλος, με τις εντολές "mkdir /users/savas/kaldi/egs/usc/data/lang" , "mkdir /users/savas/kaldi/egs/usc/data/local" , δημιουργούμε αρχικά τους υπο-φακέλους lang και local ,αντίστοιχα, εντός του φακέλου data. Στην συνέχεια, κάνουμε χρήση των εντολών "mkdir /users/savas/kaldi/egs/usc/data/local/dict" , "mkdir /users/savas/kaldi/egs/usc/data/local/lm_tmp" , "mkdir /users/savas/kaldi/egs/usc/data/local/nist_lm" για την δημιουργία των φακέλων dict , lm_tmp , nist_lm μέσα στον φάκελο local που δημιουργήσαμε παραπάνω. 

# In[250]:


get_ipython().run_cell_magic('bash', '', '\n#4.1.1\ncp -p /users/savas/kaldi/egs/wsj/s5/path.sh /users/savas/kaldi/egs/usc\ncp -p /users/savas/kaldi/egs/wsj/s5/cmd.sh /users/savas/kaldi/egs/usc\n#Manually change the value of $KALDI_ROOT to path/to/kaldi_directory\n#Also we set the variables train_cmd, decode_cmd, cuda_cmd to "run.pl"\n\n#4.1.2 \n#Creation of Softlinks \n\nln -s /users/savas/kaldi/egs/wsj/s5/steps /users/savas/kaldi/egs/usc\nln -s /users/savas/kaldi/egs/wsj/s5/utils /users/savas/kaldi/egs/usc\n\n#4.1.3\n#Creation of folder "local", which contains a soft link to score_kaldi.sh\n\nmkdir /users/savas/kaldi/egs/usc/local\nln -s /users/savas/kaldi/egs/wsj/s5/steps/score_kaldi.sh /users/savas/kaldi/egs/usc/local\n\n#4.1.4\n#Creation of folder "conf" containig the modified mfcc.conf file \n\nmkdir /users/savas/kaldi/egs/usc/conf\ncp -p /users/savas/downloads/lab2_help_scripts/mfcc.conf /users/savas/kaldi/egs/usc/conf\n\n#4.1.5 \n#Creation of some extra folders \n\nmkdir /users/savas/kaldi/egs/usc/data/lang\nmkdir /users/savas/kaldi/egs/usc/data/local\nmkdir /users/savas/kaldi/egs/usc/data/local/dict\nmkdir /users/savas/kaldi/egs/usc/data/local/lm_tmp\nmkdir /users/savas/kaldi/egs/usc/data/local/nist_lm')


# ## 4.2 Προετοιμασία γλωσσικού μοντέλου
# 
# #### 1. 
# Δημιουργούμε αρχικά , τα αρχεία 'silenece_phones.txt' και 'optional_silence.txt' . Το αρχείο 'optional_silence.txt' περιέχει μόνο το φώνημα sil , ενώ το αρχείο 'silenece_phones.txt'  περιέχει το αρχείο sil καθώς και ένα βοηθητικό φώνημα spn το οποίο στην συνέχεια , κατά την δημιουργία του αρχείο lexicon.txt , θα αντιστοιχιστεί στο σύμβολο < unk >  . Το σύμβολο αυτό θα χρήσιμοποιηθεί για την περιγραφή κάθε oov φωνήματος κατα την δημιουργία του γλωσσικού μας μοντέλου . 

# In[189]:


#Creation of 'silenece_phones.txt'and 'optional_silence.txt' within data/local/dict folder
f = open(global_path+'/local/dict/silence_phones.txt','w+')
f.write('sil\n'+ 'spn\n')
f.close()
f = open(global_path+'/local/dict/optional_silence.txt','w+')
f.write('sil\n')
f.close()


# Στην συνέχεια, στο directory data/local/dict δημιουργούμε το αρχείο lexicon.txt ,το οποίο δεν έχει καμία σχέση με αυτό που παρέχεται στα δεδομένα προς επεξεργασία. Αρχικά, διαβάζουμε και αποθηκεύουμε κάθε γραμμή του αρχείο "values_lexicon.txt" σε μια λίστα που ονομάζουμε 'values' . Επιπλέον, ορίζουμε δύο νέες λίστες την temp που θα χρησιμοποιηθεί ως βοηθητική και την phones η οποία τελικά θα είναι αυτή που θα γραφεί στο αρχείο lexicon.txt . Στην συνέχεια, καλούμε την εντολή split σε  κάθε πρόταση ,που είναι αποθηκευμένη ως στοιχείο της λίστας values, παράγοντας έτσι για κάθε πρόταση μια λίστα των λέξεων από τις οποίες αποτελείται. Κάθε τέτοια λίστα που παράγεται την εισάγουμε στην λίστα temp με την εντολή append.  Έπειτα , επαναληπτικά , εξετάζουμε κάθε στοιχείο των υπολιστών της λίστας temp , αν το φώνημα υπό εξέταση βρίσκεται ήδη στην λίστα phones τότε το απορρίπτουμε , διαφορετικά το εισάγουμε σε αυτήν. Από την λίστα phones, η οποία περιέχει από μια φορά κάθε φώνημα που εμφανίζεται στα σύνολο των δεδομένων μας, κρατάμε μόνο αυτά τα οποία δεν περιέχονται στο αρχέιο silence_phones.txt και τα αποθηκεύουμε στην λίστα non_silence_phones, την οποία και ταξινομούμε. Έπειτα, γράφουμε την λίστα non_silence_phones στο αρχείο nonsilence_phones.txt. Στην λίστα phones  εισάγουμε και το φώνημα spn και την ταξινομούμε . Τέλος, το αρχείο lexicon.txt γράφεται ως εξής phone phone , όπου phone είναι το κάθε φώνημα που περιέχεται στην λίστα phones. Σημείωνουμε, πως εισάγουμε τον ενδεχόμενο το phone να είναι το spn όπου σε αυτή την περίπτωση στο αρχείο lexicon.txt  θα γραφεί " spn   < unk> " , το οποίο σημαίνει πως κάθε άγνωστη λέξη θα αντιστοιχίζεται μέσω του φωνήματος spn στο σύμβολο  < unk> . 

# In[190]:


#Creation of 'lexicon.txt' within data/local/dict folder
f6 = open("./slp_lab2_data/values_lexicon.txt","r")
values = f6.readlines()
f6.close()
phones = []
temp =[]
for sentence in values :
    temp.append(sentence.split())
for i in temp :
    for item in i  :
        if item not in phones:
            phones.append(item)
non_silence_phones = phones[0:len(phones)-3]
non_silence_phones.sort()

f = open(global_path+'/local/dict/nonsilence_phones.txt','w+')
g = open(global_path+'/local/dict/lexicon.txt','w+')
for phone in non_silence_phones :
    f.write(phone+'\n')
phones = non_silence_phones
phones.append('sil')
phones.sort()
phones.insert(0,'spn')
for phone in phones :
    if phone == 'spn' :
        g.write('<unk>'+' '+phone+'\n')
    else:
        g.write(phone+' '+phone+'\n') 
f.close()
g.close()


# Συνεχίζοντας, δημιουργούμε τα αρχεία 'lm_train.text', 'lm_test.text' , 'lm_dev.text' στον φάκελο data/local/dict. Συγκεκριμένα , διαβάζουμε και αποθηκεύουμε κάθε γραμμή του αρχείου text ,που δημιουργήσαμε παραπάνω, στην λίστα to_transform. Παράλληλα, ορίζουμε την λίστα final με μήκος όσο και αυτό της λίστας to_transform. Στην συνέχεια, για κάθε στοιχείο της λίστας to_transform αντικάθιστούμε τον χαρακτήρα "\n" με τον χαρακτήρα του κενού και έπειτα αντιγράφουμε το to_transform[i] στοχείο στο final[i] μέχρι να εντοπίσουμε το φώνημα 'sil' . Μόλις εντοπίσουμε το συγκεκριμένο φώνημα εισάγουμε τον χαρακτήρα < s > . Μετά την εισαγωγή του  < s > συνεχίζουμε την αντιγραφή του to_transform[i] ,από εκεί που είχαμε μείνει, στο final[i]. Αφού ολοκληρώσουμε την αντιγραφή των φωνημάτων από το to_transform[i], εισάγουμε στο final[i] το χαρακτήρα < /s >. Έτσι, επιτυγχάνουμε την εισαγωγή των χαρακτήρων < s > και < /s > , τα οποία σηματοδοτούν την αρχή και το τέλος μια πρότασης ,αντίστοιχα, σε κάθε ακολουθία φωνημάτων.  Τέλος, δημιουργούμε και το κενό αρχείο extra_questions.txt στο φάκελο data/local/dict. 

# In[191]:


#Creation of 'lm_train.text', 'lm_test.text' 'lm_dev.text' within data/local/dict folder
for i in paths:
    f = open(global_path+'/local/dict/lm_{}.text'.format(i),'w+')
    g = open(paths[i]+'/text','r')
    to_transform = g.readlines()
    final = ['']*len(to_transform)
    for j in range(0,len(to_transform)) :
        to_transform[j] = to_transform[j].replace('\n',' ')
        final[j]+=to_transform[j][0:to_transform[j].index('sil')]+'<s> '+to_transform[j][to_transform[j].index('sil'):]+'</s>\n' 
    for stuff in final :
        f.write(stuff)
    f.close()
    g.close()
#Creation of 'extra_questions.txt' within data/local/dict folder
f = open(global_path+'/local/dict/extra_questions.txt',"w+")
f.close()


# #### 2 - 3. 
# • Γίνεται χρήση της εντολής build-lm.sh του πακέτου IRSTLM. Το script κανει split την διαδικασία της         εκτίμησης σε 1 και 2 jobs αντίστοιχα για τους φακέλους test, train και dev αντίστοιχα, με σκοπό την
# δημιουργία των unigram και bigram μοντέλων αντίστοιχα
# 
# • Το script παράγει ένα αρχείο LM στην μορφή .ilm.gz που δεν είναι η τελική ARPA μορφή, αλλά
# μία ενδιάμεση που ονομάζεται iARPA και η οποία αναγνωρίζεται και απαιτείται από την compile-lm.sh
# ,καθώς και από τον αποκωδικοποιητή Moses SMT που τρέχει με το πακέτο IRSTLM.
# 
# Κατασκευάζουμε τόσο unigram , όσο και bigram γλωσσικά μοντέλα για ta trainnig dataset ,dev dataset , καθώς και test dataset 

# In[203]:


get_ipython().run_cell_magic('bash', '', '\nsource /users/savas/kaldi/egs/usc/path.sh\nexport IRSTLM=/Users/Savas/kaldi/tools/irstlm\n\n#Build an intermediate form of unigram and bigram language model for the trainning set \n/Users/Savas/kaldi/tools/irstlm/bin/build-lm.sh -i /Users/Savas/kaldi/egs/usc/data/local/dict/lm_train.text -n 1 -o /Users/Savas/kaldi/egs/usc/data/local/lm_tmp/lm_train_uni.ilm.gz -k 5\n/Users/Savas/kaldi/tools/irstlm/bin/build-lm.sh -i /Users/Savas/kaldi/egs/usc/data/local/dict/lm_train.text -n 2 -o /Users/Savas/kaldi/egs/usc/data/local/lm_tmp/lm_train_bi.ilm.gz -k 5\n\n#Compile the intermediate form of both unigram and bigram LMs of the trainning set to their final forms \ncompile-lm /Users/Savas/kaldi/egs/usc/data/local/lm_tmp/lm_train_uni.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > /Users/Savas/kaldi/egs/usc/data/local/nist_lm/lm_train_uni.arpa.gz\ncompile-lm /Users/Savas/kaldi/egs/usc/data/local/lm_tmp/lm_train_bi.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > /Users/Savas/kaldi/egs/usc/data/local/nist_lm/lm_train_bi.arpa.gz\n\n#Build an intermediate form of unigram and bigram language model for the test set \n/Users/Savas/kaldi/tools/irstlm/bin/build-lm.sh -i /Users/Savas/kaldi/egs/usc/data/local/dict/lm_test.text -n 1 -o /Users/Savas/kaldi/egs/usc/data/local/lm_tmp/lm_test_uni.ilm.gz -k 5\n/Users/Savas/kaldi/tools/irstlm/bin/build-lm.sh -i /Users/Savas/kaldi/egs/usc/data/local/dict/lm_test.text -n 2 -o /Users/Savas/kaldi/egs/usc/data/local/lm_tmp/lm_test_bi.ilm.gz -k 5\n\n#Compile the intermediate form of both unigram and bigram LMs of the test set to their final forms \ncompile-lm /Users/Savas/kaldi/egs/usc/data/local/lm_tmp/lm_test_uni.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > /Users/Savas/kaldi/egs/usc/data/local/nist_lm/lm_test_uni.arpa.gz\ncompile-lm /Users/Savas/kaldi/egs/usc/data/local/lm_tmp/lm_test_bi.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > /Users/Savas/kaldi/egs/usc/data/local/nist_lm/lm_test_bi.arpa.gz\n\n#Build an intermediate form of unigram and bigram language model for the validation set \n/Users/Savas/kaldi/tools/irstlm/bin/build-lm.sh -i /Users/Savas/kaldi/egs/usc/data/local/dict/lm_dev.text -n 1 -o /Users/Savas/kaldi/egs/usc/data/local/lm_tmp/lm_dev_uni.ilm.gz -k 5\n/Users/Savas/kaldi/tools/irstlm/bin/build-lm.sh -i /Users/Savas/kaldi/egs/usc/data/local/dict/lm_dev.text -n 2 -o /Users/Savas/kaldi/egs/usc/data/local/lm_tmp/lm_dev_bi.ilm.gz -k 5\n\n#Compile the intermediate form of both unigram and bigram LMs of the validation set to their final forms \ncompile-lm /Users/Savas/kaldi/egs/usc/data/local/lm_tmp/lm_dev_uni.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > /Users/Savas/kaldi/egs/usc/data/local/nist_lm/lm_dev_uni.arpa.gz\ncompile-lm /Users/Savas/kaldi/egs/usc/data/local/lm_tmp/lm_dev_bi.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > /Users/Savas/kaldi/egs/usc/data/local/nist_lm/lm_dev_bi.arpa.gz')


# ##### 4. 
# Γίνεται εκτέλεση της εντολής prepare lang.sh με ορίσματα LM arpa αρχεία πλέον και των 6 μοντέλων για παραγωγή των αρχείων ”words.txt”,”oov.txt και ”phones.txt”, που περιέχουν αντίστοιχα:
# • Μία λίστα από όλες τις λέξεις στο vocabulary, επιπρόσθετα του sil και #0 συμβόλων (χρήση ως μετάβαση ε στο input του G.fst). Κάθε λέξη έχει μοναδικό index
# • Μία λίστα από όλα τα φωνήματα στο vocabulary,
# •  ́Εχει μία μόνο γραμμή με την λέξη (όχι το φώνημα) για τα εκτός vocabulary αντικείμενα. Εδώ χρησιμοποιείται το "oov" γιατί αυτό παίρνουμε από το πακέτο IRSTLM στα γλωσσσικά μοντέλα μας.
# 
# Ακόμα δημιουργούνται δύο fst. Το πρώτο είναι το ”L.fst”, ένας πεπερασμένων καταστάσεων μετατροπέας από το λεξικό, με σύμβολα-φωνημάτα στην είσοδο και σύμβολα-λέξεις στην έξοδο. Το L κάνει mapping monophone ακολουθίες σε λέξεις. Το δεύτερο είναι το ”L_disambig.fst”, το φωνητικό λεξικό για αποσαφήνιση των ασαφειών ,διφορούμενων συμβόλων-φωνημάτων, τα οποία τα χρειαζόμαστε όταν έχουμε μία λέξη που είναι prefix μιας άλλης (πχ. cat και cats στο ίδιο λεξικό). Εάν δεν τα έχεις, τότε τα μοντέλα γίνονται μη ντετερμινιστικά.

# In[211]:


get_ipython().run_cell_magic('bash', '', "source /users/savas/kaldi/egs/usc/path.sh\ncd /users/savas/kaldi/egs/usc\n#We construct our language's lexicon, represented by an fst (L.fst)\nutils/prepare_lang.sh  /users/savas/kaldi/egs/usc/data/local/dict  '<unk>'  /users/savas/kaldi/egs/usc/data/local/tmp  /users/savas/kaldi/egs/usc/data/lang")


# Τα G.fst, μπορούν να δομηθούν αποκλειστικά πάνω στα unigram και bigram μοντέλα μας. Για την διαδικασσία χρησιμοποιείται η τροποποιημένη timit_format_data.sh του kaldi , το οποίο το ονομάζουμε timit_format_data1.sh . Πιο συγκεκριμένα, δημιουργούμε το αντίγραφο ”lang_test_i_j”, όπου i = {train,dev,test} και j = {uni,bi}, που περιέχει τα αρχεία και τα αυτόματα του προηγούμενου βήματος. Mέσα σε κάθε φάκελο που δημιουργεί το script ,καλούνται οι εντολές gunzip -c και arpa2fst, επαναληπτικά και για 6 γλωσσικά μοντέλα, με σκοπό την δημιουργία των αντίστοιχων γραμμτικών σε μορφή "G.fst". Η arpa2fst μάλιστα δέχεται όρισμα την λίστα ”words.txt”, και λογικό αφού θέλουμε πληροφορία για όλες τις δυνατές λέξεις και φωνημάτα που έχουν προκύψει από όλες τις προτάσεις (άρα για αυτό έγινε εξέταση όλων train,test,dev, αφού συνολικά καλύπτουν 100% της πληροφορίας)

# In[ ]:


get_ipython().run_cell_magic('bash', '', "source /users/savas/kaldi/egs/usc/path.sh\ncd /users/savas/kaldi/egs/usc\n\n#We construct our language's grammar, represented by an fst (G.fst) for both unigram and bigram LMs\n./timit_format_data1.sh")


# Επιπλέον, δημιουργούμε από κάθε αρχείο utt2spk το αντίστοιχο spk2utt με την κλήση του κώδικα pearl  utt2spk_to_spk2utt.pl, το οποίο θα χρειαστεί παρακάτω. 

# In[212]:


get_ipython().run_cell_magic('bash', '', 'source /users/savas/kaldi/egs/usc/path.sh\ncd /users/savas/kaldi/egs/usc\n\n#Create the spk2utt file from the utt2spk one for all the datasets \nutils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt\nutils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt\nutils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt')


# ### Ερώτημα 1
# 
# • Η αξιολόγηση γλωσσικών μοντέλων περιλαμβάνει και τον υπολογισμό perplexity, εντροπίας και out-of-
# vocabulary rate τόσο του test set όσο και του validation set.
# 
# • Μεγάλα LM αρχεία μπορούν να "περιοριστούν" με έξυπνο τρόπο με την εντολή που αναφέραμε και νωρίτερα, prune, η οποία αφαιρεί n-gram μοντέλα τα οποία για την προσφυγή σε back-off αποτελέσματα οδηγούν σε μικρές-μηδαμινές απώλειες. Η εντολή αυτή δέχεται ως όρισμα ένα threshold,το οποίο έχει νόημα γα τα bigram μοντέλα του test και του validation set και το οποίο βέβαια εχει αξία για εμπειρικά δεδομένα. (threshold = 0 δεν οδηγεί σε περιορισμό).
# 
# • Η εφαμοργή του pruning οδηγεί σε LM αρχεία μορφής ”.plm” που περιέχει λιγότερα bigrams, για τις προδιαγραφές του παρόντος εργαστηρίου. Το output είναι ένα ARPA LM αρχείο. Με σκοπό να μετρήσουμε την απώλεια της ακρίβειας που εισήχθη με το pruning, το perplexity του output LM μπορεί να υπολογιστεί (δεν θα διαφέρει πολύ από το αντίστοιχο LM χωρίς εφαρμογή pruning).
# 
# • Τα LM είναι όντως χρήσιμα και έχουν αξία όταν έχουμε αρκετά δεδομένα στην διάθεση μας, και έτσι μπορούμε να απευθύνουμε queries σε αυτά για υπολογισμό peplexity που έχει ουσία. Συγκεκριμένα, οι προδιαγραφές του εργαστήριου προβλέπουν αρκετά δεδομένα, και για αυτόν ακριβώς τον λόγο ΠΡΕΠΕΙ να παρατηρηθεί ότι το perplexity των bigram μοντέλων είναι χαμηλότερο από το αντίστοιχο των unigram μοντέλων τόσο για το test set όσο και για το validation set.
# 
# • Υπολογίζεται με την εντολή compile-lm, αλλά αυτή την φορά με όρισμα –eval=test. Παρακάτω αναδεικνύον- ται τα αποτελέσματα.

# In[269]:


get_ipython().run_cell_magic('bash', '', 'source /users/savas/kaldi/egs/usc/path.sh\ncd /users/savas/kaldi/egs/usc/data/local/lm_tmp\nexport IRSTLM=$KALDI_ROOT/tools/irstlm/\nexport PATH=${PATH}:$IRSTLM/bin\n\n\necho "########################################################"\necho "Calculating perplexity for evaluation data unigram model"\necho "########################################################"\n\nprune-lm --threshold=1e-6,1e-6 lm_dev_uni.ilm.gz lm_dev_uni.plm\ncompile-lm lm_dev_uni.plm --eval=../dict/lm_dev.text --dub=10000000\necho""\necho "#######################################################"\necho "Calculating perplexity for evaluation data bigram model"\necho "#######################################################"\nprune-lm --threshold=1e-6,1e-6 lm_dev_bi.ilm.gz lm_dev_bi.plm\ncompile-lm lm_dev_bi.plm --eval=../dict/lm_dev.text --dub=10000000\necho""\necho "##################################################"\necho "Calculating perplexity for test data unigram model"\necho "##################################################"\n\nprune-lm --threshold=1e-6,1e-6 lm_test_uni.ilm.gz lm_test_uni.plm\ncompile-lm lm_test_uni.plm --eval=../dict/lm_test.text --dub=10000000\necho""\necho "#################################################"\necho "Calculating perplexity for test data bigram model"\necho "#################################################"\nprune-lm --threshold=1e-6,1e-6 lm_test_bi.ilm.gz lm_test_bi.plm\ncompile-lm lm_test_bi.plm --eval=../dict/lm_test.text --dub=10000000')


# Επομένως παρατηρείτει πτώση από 40.13 σε 18.24 στο unigram μοντέλο και από 39.75 σε 17.61 στο bigram μοντέλο  για το validation και το test set αντίστοιχα, πράγμα απόλυτα δικαιολογημένο, αφού όσο χαμηλότερο είναι τόσο πιο αποτελεσματική κατανομή πιθανοτήτων έχουμε για πρόβλεψη δειγμάτων.

# ## 4.3 Εξαγωγή ακουστικών χαρακτηριστικών
# 
# Η διαδικασσία εξαγωγής features αναφέρεται και ως παραμετροποίηση ομιλίας, δηλαδή η ανάλυση των σημάτων ήχου σε μικρή κλίμακα για στατιστικούς σκοπούς και για κατηγοριοποίηση των φασματικών features των σημάτων με σκοπό την προετοιμασία τους για την μετέπειτα αποκωδικοποίηση. Ουσιαστικά η διαδικασία αυτή ”γεφυρώνει” πληροφορίες ήχου και στατιστικών μοντέλων ASR.
# 
# Οι δύο πιο γνωστές μορφές είναι MFCC(Mel Frequency Cepstral Coefficients) και PLP (Perceptual Linear Prediction) με την πρώτη μορφή να λαμβάνει υπόψιν την φύση της φωνής ,ενώ με το LPC γίνεται πρόβλεψη μελλοντικών features βάσει προηγούμενων. Εφόσον η ανθρώπινη φωνή και το ακουστικό σύστημα του ανθρώπου είναι μη γραμμικά, το LPC δεν είναι καλή επιλογή για εκτίμηση ακουστικών features. Αντίθετα, τα MFCC βασίζονται στην λογική των λογαριθμικά διαχωρισμένων φίλτρων και για αυτό θεωρούνται αποτελεσματικότερα.
# 
# Τώρα θα παράγουμε τα features και τα αντίστοιχα ”feats.scp” που κάνει mapping τα utterance ids στις θέσεις ενός αρχείου, εδώ ”feats.ark.”. Για GMM-HMM συστήματα, τυπικά χρησιμοποιούμε MFCC ή PLP features και μετά εφαρμόζουμε cepstral mean και variance normalisation
# 
# Εδώ θα παράξουμε MFCCs -για τα οποία δίνουμε μια σχετική θεωρητική ανάλυση στην αρχή της παρούσας αναφοράς- για τα δεδομένα μας. Συγκεκριμένα, για να αντισταθμίσουμε την μεταβολή speech και speaker, η εφαρμογή της παραπάνω μεθόδου στα MFCC κρίνεται αναγκαία.
# 
# Εκτελούμε τις εντολές kaldi make_mfcc.sh και compute_cmvn_stats.sh, για την παραγωγή των όσων αναφέρθηκαν,για κάθε ένα από τα τρία directory ξεχωριστά. .Τα τελικά αποτελέσματα περιέχονται στους φακέλους ”/data/*/data" , όπου * ={train,dec,test} , σε μορφή ”.ark”, ενώ εκείνα των στατιστικών αναλύσεων στον φάκελο ”/data/*/data/log",όπου * ={train,dec,test}, σε μορφή ”.log”

# In[213]:


get_ipython().run_cell_magic('bash', '', 'source /users/savas/kaldi/egs/usc/path.sh\ncd /users/savas/kaldi/egs/usc\n\n#Computation of MFFCS \nsteps/make_mfcc.sh data/train\nsteps/make_mfcc.sh data/test\nsteps/make_mfcc.sh data/dev\n\n#Compute the Cepstral Means and perform their Variance Normalization\nsteps/compute_cmvn_stats.sh data/train\nsteps/compute_cmvn_stats.sh data/test\nsteps/compute_cmvn_stats.sh data/dev')


# ### Ερώτημα 2
# 
# Στην αναγνώριση φωνής χρειάζεται να αφαιρέσουμε οποιοδήποτε σήμα περιπλέκεται με το σήμα του ομιλητή καθώς δεν μπορούμε να εξάγουμε καθαρά και έγκυρα τις παραμέτρους . Τέτοια σήματα μπορεί να προέρχονται από το περιβάλλον του ομιλητή ,όπως θόρυβος, άλλες ομιλίες και εν γένη άλλες ηχητικές πηγές. ́Εχοντας το σήμα εισόδου x[n] το οποίο περνάει από φίλτρο απόκρισης h[n] παίρνουμε ένα σήμα εξόδου y[n]=x[n]*h[n] το οποίο είναι η συνέλιξη των x[n] και h[n].Χρησιμοποιώντας την ιδιότητα του Μ/Σ Fourier ισχύει ότι: Y [f] = X[f]H[f]
# 
# Σαν επόμενο βήμα παίρνουμε cepstrum μέσω του λογαρίθμου του φάσματος:Y[q] = log(X[f]H[F]) = log(X[f]) + log(Y [f]) = X[q] + H[q]
#  
# Παρατήρηση: Από την συνέλιξη στο πεδίο του χρόνου μεταβήκαμε στον πολλαπλασιασμό στο πεδίο της συχνότητας και στην πρόσθεση στο πεδίο του cepsrtum. Τα παραπάνω βήματα θα χρειαστούν για τον υπολογισμό του Cepstral Mean Normalization.
#  
# Πλέον με τις μετατροπές αυτές γνωρίζουμε ότι οποιεσδήποτε αλλοιώσεις έχουμε στο πεδίο του χρόνου από ανεπιθύμητους παράγοντες στο πεδίο του cepstrum τους έχουμε σε μορφή αθροίσματος Κάνοντας λοιπόν την υπόθεση ότι όλα αυτά είναι στατικά(stationary) μπορούμε να θωρήσουμε ότι το i-οστό πλαίσιο έχει την μορφή: Yi[q] = H[q] + Xi[q].
# 
# Παίρνοντας τον μέσο όρο των πλαισίων του συνόλου των πλαισίων έχουμε : N1 􏰀i Yi[q] = H[q]+ N1 􏰀i Xi[q]
# Τώρα ορίζουμε την διαφορά “Ri”του i-οστού πλαισίου ως αυτήν του μέσου όρου ,που υπολογίσαμε πριν
#  ,από την έξοδο: Ri[q] = Yi[q] − 1 􏰀 Yi[q] = H[q] + Xi[q] − (H[q] + 1 􏰀 Xi[q]). Επομένως: Ri[q] =
# 1􏰀NiNi
# Xi[q] που εν τέλη καταλήγει να είναι η διαφορά του Xi από τον μέσο όρο όλων των παραθύρων
# 
# Το Cepstral Mean Normalization δεν είναι υποχρεωτικό ,ειδικά όταν πρόκειται για έναν ομιλητή σε ένα φυσιολογικό περιβάλλον χωρίς εξαιρετικό θόρυβο. Ωστόσο κρίνεται εξαιρετικό εργαλείο για την εξαγωγή καλύτερων καλύτερων ποσοστών με τον συνδυασμό ενός κέρδους(gain) πάνω στο σήμα καθώς τότε τα ποσοστά επιτυχούς αναγνώρισης είναι σημαντικά καλύτερα . Παρατήρηση:  ́Εχει αποδειχθεί ότι η μέθοδος CMVN είναι ιδιαίτερα αποτελεσματική για μικρές εκφράσεις(utterances).

# ### Ερώτημα 3
# 
# Γίνεται εκτέλεση του παρακάτω bash script ,στο οποίο εκτελείται η εντολή feat-to-dim η οποία
# επιστρέφει την διάσταση των MFCC για το πρώτο από τα 4 sections που έχουμε κάνει split Dimension = 13. Σημειώνεται πως το dimension αυτό θα είναι ίδιο για όλα τα features του set
# 
# Επιπλέον ,γίνεται εκτέλεση της feat-to-len η οποία επιστρέφει τον αριθμό των frames για κάθε ένα από τα 46 utterrances id’s στο πρώτο από τα 4 section, εμείς θέλουμε τα 5 πρώτα. Number of Features = [371,336,519,400,397] για τα utterance_ids = [4,16,43,48,53]. Η αντιστοιχία από το utterance_id σε αυτό τον πίνακα που γράψαμε παραπάνω γίνεται με την βοήθεια του αρχείο uttids που φτιάξαμε κατα την προπαρασκευή.

# In[292]:


get_ipython().run_cell_magic('bash', '', 'source /users/savas/kaldi/egs/usc/path.sh\n\nfeat-to-dim ark:/users/savas/kaldi/egs/usc/data/train/data/raw_mfcc_train.1.ark -')


# In[296]:


get_ipython().run_cell_magic('bash', '', 'source /users/savas/kaldi/egs/usc/path.sh\n\nfeat-to-len scp:/users/savas/kaldi/egs/usc/data/test/feats.scp ark,t:/users/savas/kaldi/egs/usc/data/test/feats.lengths\ncat /users/savas/kaldi/egs/usc/data/test/feats.lengths')


# ## 4.4 Εκπαίδευση ακουστικών μοντέλων και αποκωδικοποίηση προτάσεων
# 
# Η αποκωδικοποιητική ικανότητα ενός συστήματος ASR εξαρτάται σχεδόν εξ’ολοκλήρου από τις επιδράσεις του γλωσσικού και του ακουστικού μοντέλου που κατασκευάσαμε. Το αντικείμενο του πρώτου είναι ο υπολογισμός της πιθανότητας μιας πρότασης στο τωρινό task αναγνώρισμης ανάμεσα σε ένα μεγάλο αριθμό προτάσεων, πράγμα που είναι κρίσιμο για την μείωση της πολυπλοκότητας της αναζήτησης των λέξεων στον αποκωδικοποιητή που θα κατασκευάσουμε. Επομένως, αναμένουμε το bigram μοντέλο που είναι πιο αποτελεσματικό από το unigram να οδηγήσει σε χαμηλότερο WER.
# 
# Το αντικείμενο του δεύτερου είναι να περιγράψει στατιστικώς τις λέξεις σε μορφή φωνημάτων. Η πιο γνωστή μέθοδος  είναι η GMM-HMM ( Gaussian Mixture Model-Hidden Markov model). Το HMM βασίζεται σε πεπερασμένα αυτόματα για την απότιμηση των πιθανοτήτων μετάβασης, έτσι ώστε να μπορούμε να υπολογίσουμε τις πιθανότητες των αντίστοιχων λέξεων που προκύπτουν απο τις ακολουθίες φωνημάτων. Οι HMM καταστάσεις τυπικώς συγκροτούν είτε ένα monophone είτε ένα triphone μοντέλο,όπου άλλοι μέθοδοι επίσης εφαρμόζονται για περαιτέρω βελτίωση του μοντέλου. Τα GMM μοντελοποιούν το output-παρατήρηση των HMM καταστάσεων με Gaussian μίγματα. Επομένως οι διάφορες HMM καταστάσεις θα μοντελοποιηθούν με Gaussian κατανομή μέσω ενός clustering δέντρου απόφασης στο βήμα της κατασκευής του monophone μοντέλου
# 
# #### 1.
# 
# Το monophone μοντέλο είναι το πρώτο κομμάτι της διαδικασίας training. Αποτελεσματικά monophone μοντέλα μπορούν να σχηματιστούν και με λίγα δεδομένα, όπως στην συγκεκριμένη περίπτωση τα 1448 του train set, κυρίως με σκοπό να δώσουν boost σε μετέπειτα μοντέλα, όπως το triphone που θα παράξουμε παρακάτω. Τα απαιτούμενα ορίσματα είναι διαρκώς κατά το training τα ακόλουθα:
# 
# -Location of the accoustic : data/train
# -Location of the lexicon : data/lang_test_train_uni or data/lang_test_train_bi (Ανάλογα το μόντελο που εκπαιδεύουμε)
# -Destination Directory : exp/mono_uni or exp/mono_bi(Ανάλογα το μόντελο που εκπαιδεύουμε)
# 
# Εκτελούμε την εντολή train_mono.sh τόσο για το unigram LM , όσο και για το bigram. Το αποτέλεσμα είναι η παραγωγή των ”mono_uni” και ”mono_bi”μοντέλων.
# 
# Ακολοθούμε ακριβώς την ίδια διαδικασία με προηγούμενως για την κατασκευή των αντίστοιχων alignments του μοντέλου με την μόνη διαφορά ότι:
# 
# Destination directory for the alignment: ”exp/mono_ali_uni” or ”exp/mono_ali_bi”

# In[253]:


get_ipython().run_cell_magic('bash', '', '\n#4.4.1\nsource /users/savas/kaldi/egs/usc/path.sh\ncd /users/savas/kaldi/egs/usc\n\n# Train and align unigram (based) monophone accoustic model \nsteps/train_mono.sh  data/train data/lang_test_train_uni exp/mono_uni\nsteps/align_si.sh data/train data/lang_test_train_uni exp/mono_uni exp/mono_ali_uni \n\n#Train and align bigram (based) monophone accoustic model\nsteps/train_mono.sh  data/train data/lang_test_train_bi exp/mono_bi\nsteps/align_si.sh data/train data/lang_test_train_bi exp/mono_bi exp/mono_ali_bi')


# ### 2.
# 
# Παρόμοιο FST framework, που διαδραμάτισε ρόλο κλειδί στην kaldi γραμματική, χρησιμοποίηθηκε τόσο για την training όσο για την testing διαδικασία. Το appendix C αποτελεί απεικόνιση του πως το Kaldi θα μπορούσε να παράξει transcriptions από ακουστικά features όσον αφορά την συγκεκριμένη γραμματική. Δημιουργήσαμε 6 τέτοιους γράφους έναν για κάθε μία από τις 6 γραμματικές (unigram και bigram των test, train και evaluation set) με εκτέλεση της εντολής utils/mkgraph.sh –mono. Τα αποθήκευσαμε στον φάκελο του γράφου graph_nosp_tgpr_uni και  graph_nosp_tgpr_bi για το mono_uni και mono_bi μοντέλο αντίστοιχα, με σκοπό την ανάκτηση τους σε περίπτωση που χρειαστεί.
# 

# In[254]:


get_ipython().run_cell_magic('bash', '', '\n#4.4.2\nsource /users/savas/kaldi/egs/usc/path.sh\ncd /users/savas/kaldi/egs/usc\n\n#Creation of HCLG graph based on unigram model \nutils/mkgraph.sh  data/lang_test_train_uni exp/mono_uni exp/mono_uni/graph_nosp_tgpr_uni\n\n#Creation of HCLG graph based on bigram model \nutils/mkgraph.sh  data/lang_test_train_bi exp/mono_bi exp/mono_bi/graph_nosp_tgpr_bi')


# ### 3.
# 
# Τελικά, αφού έχουμε ολοκληρώσει το μοντέλο μας, ήρθε η ώρα να το εφαρμόσουμε στo test και evaluation set που μας δίνεται. Το μοντέλο μας, στα νέα πλέον δεδομένα, προσπαθεί να κάνει την κατάλληλη αποκωδικοποίηση και τα αποτελέσματα βρίσκονται στα αρχεία wer πoυ βρίσκονται στο path ”exp/mono_j/decode_i_j”, όπου i={dev,test} και j={uni,bi}.
# 
# Εκτελούμε επαναληπτικά τον αλγόριθμο 6 φορές, μία για κάθε HCLG γράφο που δημιουργήσαμε στο προηγούμενο βήμα. Πιο συγκεκριμένα, εκτελούμε την εντολή steps/decode με –nj = 4 (αριθμός pipelines αναθετημένα σε 4 CPU πυρήνες), με ορίσματα τον γράφο-αποκωδικοποιητή και τον φάκελο data/i ,όπου i = {dev,test}, δηλαδή τα utterance συσχετιζόμενα αρχεία.
# 
# Το βάρος του γλωσσικού μοντέλου για το οποίο ο αποκωδιποιητής θα ψάξει, επιβεβαιώνετε από την ”κρυφή” και ταυτόχρονη εκτέλεση του score kaldi.sh (με –scoring-opts από 1 έως 20) για τον υπολογισμό του PER 
# 
# Η αποκωδικοποίηση περιλαμβάνει λοιπόν wer και log αποτελέσματα στους φακέλους ”decode_dev_i” και ”decode_test_i ” του φακέλου ”mono_i” , όπου i={uni,bi}

# In[255]:


get_ipython().run_cell_magic('bash', '', '\n#4.4.3\nsource /users/savas/kaldi/egs/usc/path.sh\ncd /users/savas/kaldi/egs/usc\n\n#Decode validation and test dataset using the HCLG graph created by unigram LM\nsteps/decode.sh --nj 4  exp/mono_uni/graph_nosp_tgpr_uni data/dev exp/mono_uni/decode_dev_uni \nsteps/decode.sh --nj 4  exp/mono_uni/graph_nosp_tgpr_uni data/test exp/mono_uni/decode_test_uni \n\n#Decode validation and test dataset using the HCLG graph created by bigram LM\nsteps/decode.sh --nj 4 exp/mono_bi/graph_nosp_tgpr_bi data/dev exp/mono_bi/decode_dev_bi \nsteps/decode.sh --nj 4 exp/mono_bi/graph_nosp_tgpr_bi data/test exp/mono_bi/decode_test_bi ')


# ### 4.
# 
# Με το που τελειώνει η αποκωδικοποίηση για την δυάδα test και validation set μετά την εφαρμογή κάποιου απο τους 6 HCLG γράφους, εκτελείται η εντολή utils/best wer.sh. για εύρεση του καλύτερου δυνατού PER (δηλαδή του μικρότερου δυνατού)
# 
# Παρακάτω φαίνονται τα αποτελέσματα του καλύτερο WER που πήραμε από την εκτέλεση για όλους τους γράφους HCLG, και άρα για όλα τα unigram και bigram γλωσσικά μοντέλα που κατασκευάσαμε:
# 

# In[257]:


get_ipython().run_cell_magic('bash', '', '\n#4.4.4\nsource /users/savas/kaldi/egs/usc/path.sh\ncd /users/savas/kaldi/egs/usc\n\n#Printing the PER  for both unigram and bigram (based) models \necho "PER for Validation dataset based on the Unigram accoustic model :"\n[ -d exp/mono_uni/decode_dev_uni ] && grep WER exp/mono_uni/decode_dev_uni/wer_* | utils/best_wer.sh\necho""\necho "PER for Test dataset based on the Unigram accoustic model :"\n[ -d exp/mono_uni/decode_test_uni ] && grep WER exp/mono_uni/decode_test_uni/wer_* | utils/best_wer.sh\necho " "\necho "PER for Validation dataset based on the Bigram accoustic model :"\n[ -d exp/mono_bi/decode_dev_bi ] && grep WER exp/mono_bi/decode_dev_bi/wer_* | utils/best_wer.sh\necho ""\necho "PER for Test dataset based on the Bigram accoustic model :"\n[ -d exp/mono_bi/decode_test_bi ] && grep WER exp/mono_bi/decode_test_bi/wer_* | utils/best_wer.sh')


# ## Ερώτημα 
# 
# Οι υπερπαράμετροι του scoring script για το συγκεκριμένο βήμα είναι το min και max LM-weight για lattice rescoring που πήραν τιμή 1 και 20 αντίστοιχα (το default-συνηθισμένο είναι 9 και 20).

# ### 5.
# 
# Το να κάνουμε train ένα triphone μοντέλο μεταφράζεται σε παραπάνω arguments για τον αριθμό των φύλλων, ή αριθμό HMM καταστάσσεων στο δέντρο απόφασης και στον αριθμό των Gaussianss.
# 
# Γίνεται εκτέλεση της steps/train deltas.sh, με σκοπό να κάνουμε train τα aligments του monophone μοντέλου σε triphone, το directory του οποίου ονομάζουμε ”tri1_uni” και ”tri1_bi”, ανάλογα με το mono που έχει χρησιμοποιηθεί για την εκπαίδευση του triphone. Θα μπορούσαμε να είχαμ μία HMM κατάσταση για κάθε φώνημα, αλλά γνωρίζουμε ότι τα φωνήματα ποικίλλουν σε σημαντικό βαθμό ανάλογα με το αν βρίσκονται στην αρχή, στην μέση ή στο τέλος μίας πρότασης. Αυτό μας οδηγεί σε τουλάχιστον 168 HMM καταστάσεις μόνο για το variation. Με 2000 HMM καταστάσεις, το μοντέλο μπορεί να αποφασίσει αν είναι καλύτερο να κατανείμει μία μοναδική HMM κατάσταση σε ποιο refined αλλόφωνα του αρχικού φωνήματος. Αυτό το split των φωνημάτων αποφασίζεται από φωνητικές ερωτήσεις στο ”questions.txt” και στο ”extra questions.txt”
# 
# Ο ακριβής αιρθμός φύλλων και Gaussian αποφασίζεται συχνά από ευρυστικές, ο οποίος εξαρτάται από την ποσότητα των δεδομένων, τον αριθμό των ερωτήσεων περί φωνημάτων και τον στόχο του μοντέλου. Επίσης υπάρχει ο περιορισμός ότι ο αριθμός των Gaussians πρέπει ΠΑΝΤΑ να ξεπερνάει τον αριθμό των φύλλων.  ́Οπως βλέπουμε στο script ”4 4.sh”, εμπειρικά ορίσαμε #Leaves = 2000 και #Gaussians = 10000.
# 
# Μετα ακολουθείται πιστά ακριβώς η ίδια διαδικασία που περιγράφηκε στο προηγούμενο βήμα για αποκωδικοποίηση του test και evaluation set βάσει του triphone πλέον μοντέλου και εύρεση όλων των σχετικών PER που παρουσιάζονται παρακάτω:

# In[259]:


get_ipython().run_cell_magic('bash', '', '\n#4.4.5\nsource /users/savas/kaldi/egs/usc/path.sh\ncd /users/savas/kaldi/egs/usc\n\n#Computes training alignments using a model with  delta+delta-delta features features.\n\nsteps/train_deltas.sh 2000 10000 data/train data/lang_test_train_uni exp/mono_ali_uni exp/tri1_uni \nsteps/train_deltas.sh 2000 10000 data/train data/lang_test_train_bi exp/mono_ali_bi exp/tri1_bi \n\nsteps/align_si.sh data/train data/lang_test_train_uni exp/tri1_uni exp/tri1_ali_uni\nsteps/align_si.sh data/train data/lang_test_train_bi exp/tri1_bi exp/tri1_ali_bi\n\n#NumLeaves = 2000: it can recognize all phone sequences, the numleaves determines \n#how many classes it splits them into for purposes of pooling similar \n#contexts. \n#NumGauss = 10000: The number of Gaussians is the total over all leaves, so the average \n#num-gauss per pdf/leaf is num-gauss/num-leaves, but leaves with more \n#data will get somewhat more Gaussians. \n\n#=============Make AGAIN the testing with the new trained model=================\n#Decoding again Validation, test with the new trained model\n#FEATURE TYPE IS DELTA\n\nutils/mkgraph.sh data/lang_test_train_uni exp/tri1_uni exp/tri1_uni/graph_nosp_tgpr_uni \nutils/mkgraph.sh data/lang_test_train_bi exp/tri1_bi exp/tri1_bi/graph_nosp_tgpr_bi \n\nsteps/decode.sh --nj 4  exp/tri1_uni/graph_nosp_tgpr_uni data/dev exp/tri1_uni/decode_dev_uni \nsteps/decode.sh --nj 4  exp/tri1_uni/graph_nosp_tgpr_uni data/test exp/tri1_uni/decode_test_uni\n\nsteps/decode.sh --nj 4  exp/tri1_bi/graph_nosp_tgpr_bi data/dev exp/tri1_bi/decode_dev_bi\nsteps/decode.sh --nj 4  exp/tri1_bi/graph_nosp_tgpr_bi data/test exp/tri1_bi/decode_test_bi  \n')


# In[262]:


get_ipython().run_cell_magic('bash', '', '\nsource /users/savas/kaldi/egs/usc/path.sh\ncd /users/savas/kaldi/egs/usc\n\necho "PER for Validation dataset based on the Unigram triphone accoustic model :"\n[ -d exp/tri1_uni/decode_dev_uni ] && grep WER exp/tri1_uni/decode_dev_uni/wer_* | utils/best_wer.sh\necho""\necho "PER for Test dataset based on the Unigram triphone accoustic model :"\n[ -d exp/tri1_uni/decode_test_uni ] && grep WER exp/tri1_uni/decode_test_uni/wer_* | utils/best_wer.sh\necho ""\necho "PER for Validation dataset based on the Bigram triphone accoustic model :"\n[ -d exp/tri1_bi/decode_dev_bi ] && grep WER exp/tri1_bi/decode_dev_bi/wer_* | utils/best_wer.sh\necho""\necho "PER for Test dataset based on the Bigram triphone accoustic model :"\n[ -d exp/tri1_bi/decode_test_bi ] && grep WER exp/tri1_bi/decode_test_bi/wer_* | utils/best_wer.sh')


# ## Ερώτημα 4 
# 
# Σκοπός ενός ακουστικού μοντέλου είναι να περιγράψει στατιστικά τις λέξεις ενός λεξιλογίου μέσω ακολουθιών φωνημάτων. Το μοντέλο GMM-HMM είναι ένα γνωστό μοντέλο αρκετά αποτελεσματικό. Το κομμάτι των HMM(Hidden Markov Model) είναι βασισμένο σε ένα αυτόματο πεπερασμένων καταστάσεων το οποίο αναπαριστά τις πιθανότητες μετάβασης φωνημάτων σε λέξεις .Το μοντέλο αυτό καταλήγει να αναπαριστά την πιθανότητα εμφάνισης λέξεων μέσω των ακολουθιών από διακριτά φωνήματα.
# 
# Οι πιο γνωστές αρχιτεκτονικές HMMs για αναγνώριση φωνής συνισ τώνται από 3 κατασ τάσεις όπως παρουσιάζεται δίπλα στις λευκές καταστάσεις. Η λειτουργία του GMM έχει σκοπό να προσδιορίσει πάνω στο αυτό- ματο και να ομαδοποιήσει καταστάσεις σε συγκεκριμένες υποκατηγορίες το οποίο λειτουργεί κάτι σαν χαρτογράφηση περιοχών πάνω στο αυτόματο του ΗΜΜ. Για παράδειγμα θα μπορεί να προσδιορίσει ένα φώνημα που εμφανίζεται σε πολλές λέξεις. Η εκπαίδευση αυτού του μοντέλου γίνεται επαναληπτικά πάνω σε ένα train set το οποίο βοηθά το GMM με βάσει τα αποτελέσματα του HMM για τις νέες εισόδους που δέχεται να κάνει ταξινόμηση σε κλάσεις.
# 
# 
# 
# ## Ερώτημα 5
# 
# Η πιθανότητα posteriori υπολογίζεται με τον ακόλουθο τύπο,όπου X είναι ένα διάνυσμα παρατηρήσεων(features) που έχει προκύψει από το ακουστικό μοντέλο βάσει των οποίων υπολογίζεται ποια είναι η πιθανότητα της λέξης W δεδομένου του Χ.
# $$P(W|X) = \frac{P(W)P(X|W)}{P(X)}$$
# 
# Για να βρούμε την πιο πιθανή λέξη (ή φώνημα ) στην δική μας περίπτωση θα χρησιμοποιήσουμε την συνάρτηση argmax πάνω στο σύνολο των λέξεων (ή φωνημάτων) και ο τύπος θα έχει την μορφή
# 
# $$W_{most_{pos}} = argmax_{_{_{W∈ω}}}P(W|X)$$

# ## Ερώτημα 6
# 
# Γενικά υπάρχουν τέσσερα επίπεδα στον γράφο-αποκωδικοποιητή και μια σύνθεση τεσσάτων γράφων-συστατικών ήταν αναμενόμενη για την κατασκευή του τελικού γράφου HCLG:
# • H - mapping από HMM μεταβάσεις σε context-dependent labels
# • C - mapping από context-dependent labels σε φωνήματα
# • L - mapping από φωνήματα σε λέξεις
# • G - το γλωσσσικό μοντέλο (input και output είναι λέξεις)
# Η γραμματική ουσιαστικά χρησιμοποιείται και στην φάση του training και στην φάση του testing. Το παράρτημα C με το περιεχόμενο των λέξεων αναπαριστά ένα παράδειγμα για το πως το KALDI μοντελοποιεί την μετάβαση από το ακουστικό μοντέλο σύμφωνα με την γραμματική του.

# In[ ]:




