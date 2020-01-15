#!/usr/bin/env python
# coding: utf-8

# # Προεργασία
# 
# ### Ονοματεπώνυμο : Σιφναίος Σάββας 
# 
# ### ΑΜ : 03116080
# 
# Στο κομμάτι της προεργασίας καλούμαστε να δημιουργήσουμε ένα corpus, πάνω στο οποίο θα εργαστούμε στην συνέχεια. Επιπλέον, κατά το προπρασκευαστικό στάδιο της εργαστηριακής αυτής άσκησης, θα κάνουμε χρήση μηχανών πεπερασμένων καταστάσεων, μέσω των οποίων θα οδηγηθούμε ,τελικά, στην κατασκευή ενός minimum distance Spell Cheker

# ## Bήμα 1ο
# 
# Στο πρώτο βήμα, καλούμαστε να δημιουργήσουμε ένα Corpus με το οποίο θα εργαστούμε στην συνέχεια. Για την παρασκευή αυτού του τελικού μας Corpus, επιλέγουμε να κατέβασουμε ένα συνόλο αποτελούμενο από 5 βιβλία(corpora), τα οποία συνδεαζόμενα σε ένα .txt αρχείο δημιουργούν το τελικό Corpus. Επιλέγουμε, να χρησιμοποιήσουμε παραπάνω από ένα corpus για την σύνθεση του τελικού (προς επεξεργασία) αρχείου, δίοτι με αυτόν τροπό επιτυγχάνουμε να αποκτήσουμε πλουσιότερο λέξικο, που θα μας φάνει χρήσιμο κατα την υλοποίηση του spell checker. Επιπλέον, αυξάνοντας τον αριθμό τον corpora , έχουμε μεγαλύτερες πιθανότητες να προκύψουν περισσότερα συγκείμενα (context) για κάθε λέξη του λεξικού μας. Τέλος, με την χρήση πολλαπλών corpora , μπορούμε να εντοπίσουμε και τις διαφορετικές ερμηνείες και χρήσεις μιας λέξης.
# 

# ## Βήμα 2ο 
# 
# Στο σήμειο αυτό, μετά την δημιουργία του Corpus, ξεκινάμε να ορίσουμε ορισμένες συναρτήσεις οι οποίες θα χρησιμοποιήθουν στην συνέχεια για την προεπεξεργασία του Corpus.
# 
# Αρχικά η identity_preprocess(s), δέχεται σαν όρισμα ένα string και επιστρέφει τον εαύτο του. Στην συνέχεια, η my_function (path,function)δέχεται σαν όρισμα το path για το σημείο στο οποίο είναι αποθηκεύμενο το Corpus, καθώς επίσης και την μία συνάρτηση. Η my_function (path,function),  διαβάζει το αρχείο που βρίσκεται στο συγκεκριμένο path και επιστρέφει το αποτέλεσμα που προκύπτει από την κλήση της function πάνω στο αρχείο του path.Επιπλέον, δημιουργούμε την συνάρτηση tokenize(s), της οποίας το όρισμα είναι μία συμβολοσειρά. Η tokenize(s), καλεί την strip () πάνω στην s και την lower() με την μέσω της οποίας μετατρέπει όλους του χαρακτήρες που περιέχονται στην s σε "μικρούς". Στην συνέχεια, βάσει του ASCII κώδικα του κάθε χαρακτήρα στην συμβολοσειρά, επιλέγει να κρατήσει μόνο του χαρακτήρες που αποτελούν γράμματα της αγγλικής γλώσσας. Τέλος, με την κλήση της split(), επιτυγχάνεται η δημιουργία μιας λίστας που περιέχει μόνο της λέξης του string σε μικρούς χαρακτήρες, απαλλαγμένες από αριθμούς και άλλα ειδικά σύμβολα. Τα στοιχεία της λίστας που επιστρέφει η tokenize(s) καλούνται tokens.

# In[51]:


from pathlib import Path
import numpy as np
import math
import subprocess


def identity_preprocess(s) :
    return s

def my_read(path, preprocess = identity_preprocess):
    fp = open(path, "r", encoding='latin-1')
    line = fp.readline()    #read first line
    line = preprocess(line) 
    text = line
    while line:
        line = fp.readline()     #read next lines
        line = preprocess(line)
        text = text + line
    fp.close()
    return text

def tokenize(s) :
    s = s.strip()
    s = s.lower()
    new_s = ''
    c=0
    for char in s:
        if((ord(char) >= 97 and ord(char) <= 122) ):
            new_s += char
            c=0
        elif ((ord(char) <= 97 or ord(char) >= 122) and c==0) :
            new_s += " "
            c = c+1
    s = new_s.split()
    return s 


    


# ## Bήμα 3ο 
# 
# Στο βήμα αυτό ορίζουμε την συνάρτηση unique(list1), η οποία δέχεται σαν όρισμα μία λίστα και επιστρέφει ένα tuple του οποίου το πρώτο στοιχείο είναι μια λίστα με τα μοναδικά στοιχεία της λίστας που δέχτηκε σαν όρισμα η συνάρτηση και το δεύτερο στοιχείο είναι μία λίστα με τους  μοναδικούς χαρακτήρες που περιέχονται στην λίστα list1. Επιπλέον, ορίζουμε και την συνάρτηση myascii(a), ή οποία δέχεται σαν όρισμα μια λίστα με τους μοναδικούς χαρακτήρες(έστω σε μια πρόταση) και δημιουργεί το αρχείο chars.syms. Το αρχείο chars.syms θα χρησιμοποιηθεί στην συνέχεια της άσκησης για την αντιστοίχισει των μοναδικών χαρακτήρων που περιέχονται στο Corpus μας σε κάποιο integer. Τέλος, η myascii(a) επιστρέφει και μια ταξινομημένη λίστα των μοναδικών χαρακτήρων.

# In[52]:


def unique(list1): 
  
    # intilize a null list 
    tokens = [] 
    alphabet = []
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in tokens: 
            tokens.append(x) 
        for char in x:
            if char not in alphabet :
                alphabet.append(char)
    return tokens , alphabet

def myascii(a):       #a is list
    anew = ['<epsilon>']
    a.sort()
    for x in a:
        anew.append(x)
    f = open("chars.syms","w+")
    for i in anew:
        f.write('%s     %d\n'%(i,anew.index(i)))
    f.close()
    return anew

corpus = my_read("corpus.txt")
s = tokenize(corpus)
(a,b)=unique(s)
tokens = a
lexicon = a 
lexicon.sort()
symbols = myascii(b)


# ## Βήμα 4ο 
# 
# Η δημιουργία του αρχείου που πραγματοποιεί την αντιστοιχεία κάθε χαρακτήρα σε έναν ακέραιο αριθμό, όπως αναφέρθηκε και παραπάνω, πραγματοποιείται με την κλήση της συνάρτησης myascii(a)

# ## Βήμα 5ο 
# α)
# Πριν την δημιουργία του Transducer ορίζουμε μία ακόμη συνάρτηση,την format_arc(src, dst, src_sym, dst_sym, w). Η παραπάνω συνάρτηση, οπώς φαίνεται, δέχεται 5 ορίσματα : 
# Το src , το οποίο αντιστοιχεί στην κατάσταση στην οποία βρίσκεται η μηχανή πεπερασμένων καταστάσεων.
# Το dst , το οποίο αντιστοιχεί στην κατάσταση στην οποία καταλήγει η μηχανή πεπερασμένων καταστάσεων, μετά από ορισμένη είσοδο.
# Το src_sym , το οποίο αντιστοιχεί στο σύμβολο το οποίο πυροδοτεί την μετάβαση από μια κατάσταση σε κάποια άλλη.
# Το dst_sym , το οποίο αντιστοιχεί στο σύμβολο στο οποίο μετατρέπεται το σύμβολο εκείνο που πυροδότησε την μετάβση(δηλαδή το src_sym)
# Το w , το οποίο αντιστοιχεί στο βάρος της ακμής μεταξύ των καταστάσεων src και dst.
# 
# Ο  Leveneshtein Transducer αποτελείται από μια μόνο κατάσταση και πραγματοποιεί 3 τύπους από edits. Το πρώτο είναι να αντιστοιχίζει κάθε χαρακτήρα στον εαύτο του με βάρος ακμής w=0. Το δεύτερο είναι η αντιστοίχιση ενός χαρακτήρα στο <epsilon> με βάρος ακμής w=1, το οποίο ουσιαστικά αντιστοιχεί σε διαγραφεί ενός χαρακτήρα. Ο τρίτος τύπος edit είναι το αντίστροφο από το προηγούμενο, δηλαδή η αντιστοίχιση του <epsilon> σε κάποιον άλλο χαρακτήρα με βάρος ακμής ,πάλι, w=1. To τελευταίο αυτό edit ισοδυναμεί με την εισαγωγή ενός νέου χαρακτήρα
#     
# Αφού έχουμε ορίσει την παραπάνω συνάρτηση, δημιουργούμε το αρχείο txt 'tran.txt',στο οποίο θα αποθηκεύσουμε τους "κανόνες" μετάβασης από μια κατάσταση σε κάποια άλλη (ή αλλίως τα edits που περιγράφησαν παραπάνω), οι οποίοι κατασκευάζουν τον transducer που υλοποιεί την Levenshtein απόσταση. Στην συνέχεια, με την εντολή "fstcompile -isymbols=chars.syms -osymbols=chars.syms tran.txt > T.fst" κάνουμε compile το αρχείο .txt σε .fst και με αυτό τον τρόπο ολοκληρώνουμε την κατασκευή του Levenshtein Transducer .
#     
# Από τη στιγμή που τα βάρη στον μετατροπέα είναι είτε μηδενικά, αν δεν αλλάζει το σύμβολο εισόδου, είτε άσοι, αν ένα σύμβολο αντικαθίσταται, εισάγεται ή διαγράφεται, παίρνουμε το shortest path αν δεν εφαρμόσουμε καμία μετατροπή στη λέξη, οπότε όλα τα βάρη μας θα είναι μηδενικά. Τότε θα πάρουμε στην έξοδο την λέξη εισόδου

# In[53]:


def format_arc(src, dst, src_sym, dst_sym, w) :
    out = '{0} {1} {2} {3} {4}\n'.format(src, dst, src_sym, dst_sym, w)
    return out

f = open('tran.txt', 'w+')
for i in symbols :
    if (i != '<epsilon>') :
        f.write(format_arc(src=0,dst=0,src_sym=i,dst_sym=i,w=0))
    for j in symbols :
        if (j != i ) :
            f.write(format_arc(src=0,dst=0,src_sym=i,dst_sym=j,w=1))
f.write('0\n')
f.close()


# In[54]:


get_ipython().run_cell_magic('bash', '', '\nfstcompile -isymbols=chars.syms -osymbols=chars.syms tran.txt > T.fst;')


# Ο συγκεκριμένος τρόπος υπολογισμού των βαρών είναι αρκετά απλοϊκός. Συγκεκριμένα, αν είχαμε στην διάθεσή μας δεδομένα λαθών τότε θα μπορούσαμε να προσαρμόσουμε τα βάρη των μετατροπών ώστε πιθανότερες μετατροπές να αντιστοιχίζονται σε μικρότερα βάρη. Αυτό μπορεί να εξαρτάται από το είδος της μετατροπής αλλά και από τον εκάστοτε χαρακτήρα. Για παράδειγμα είναι πολύ πιθανότερο να γράψουμε 'σ' αντί για 'α' από το γράψουμε 'π' αντί για α λόγω της σχετικής τους θέσης στο πληκτρολόγιο. Έτσι πρέπει να αντιστοιχίσουμε μικρότερο βάρος στην πρώτη μετατροπή ώστε να επιλέγεται ευκολότερα από την δεύτερη μέσω του συντομότερου μονοπατιού. Επίσης η συχνότητα μίας λέξης μπορεί να ληφθεί υπόψην, καθώς αν η απόσταση της λανθασμένης λέξης μεταξύ δύο λέξεων του λεξικού είναι παρόμοια, τότε είναι πιθανότερο να αντιστοιχίζεται στην λέξη που εμφανίζεται πιο συχνά.

#  ## Βήμα 6ο
#  
#  α)
#  Στο σημείο αυτό καλούμαστε να κατασκευάσουμε έναν αποδεχέα για κάθε λέξη που περιέχεται στο λεξικό. Όπως και στην περίπτωση του Transducer, έτσι και έδω θα γίνει χρήση της συνάρτησης formart_arc για την δημιουργία του αρχείου, το οποίο θα μεταφραστεί προκειμένου να δημιουργηθεί το .fst αρχείο που θα υλοποιεί τον αποδοχέα μας. Για την κατασκευή του acceptor εργαζόμαστε ώς εξής : 
# Αρχικά, δημιουργούμε ένα αρχείο .txt  στο οποίο θα γράφουμε τις εξόδους της συνάρτησεις format_arc
# Στην συνέχεια, επαναληπτικά, εξετάζουμε την κάθε λέξη του λεξικού μας φροντίζοντας πάντα η πρώτη ακμή για την κάθε λέξη να ξεκινάει από την κατάσταση μηδέν. Σημείωνουμε στο σημείο αυτό, πως για να έχουμε γνωση του ποιά είναι η κατάσταση στην οποία πρέπει να μεταβούμε κάθε φορά που εξετάζουμε καινούργια λέξη, κάνουμε χρήση της μεταβλητής cnt στην οποία προσθέτουμε στο τέλος κάθε επανάληψης την τιμή της κατάστασης στην οποία έφτασα η προηγούμενη λέξη. Τέλος, όπως κάναμε και στον Transducer με την εντολή 'fstcompile -isymbols=chars.syms -osymbols=chars.syms fsa.txt > fsa.fst' κάνουμε compile to .txt αρχείο σε .fst  και ολοκληρώνουμε την δημιουργία του Acceptor.
# 
# β) 
# Συνεχίζοντας, μας ζητείται να καλέσουμε τις εντολές fstrmepsilon , fstdeterminize, fstminimize. Η πρώτη εκ των τριών εντολών είναι υπεύθυνη για την αφαίρεση οποίασδηποτε εψίλον μετάβασης.Συγκεκριμένα, αν ο αρχικος αποδοχέας περιείχε μεταβάσεις που πυροδοτούνταν από το <epsilon>, μετά την κλήση αυτής της εντολής οι αντίστοιχες ακμές θα είχαν αφαιρεθεί. Επιπλέον, η fstdeterminize φροντίζει για την μετατροπή μιας μη ντερμινιστικής μηχανής στην ισοδύναμη ντετερμινιστική της. Τέλος, με την εφαρμογή  της fstminize πανώ στο ντετερμινιστικό ,πλέον, αποδοχέα , επιτυγχάνουμε την ελαχιστοποίηση της μηχανής αυτής. Με την εκτέλεση των τριών αυτών εντολών επιτυγχάνουμε σημαντική μείωση του αριθμού των καταστάσεων της μηχανής,όπως φαίνεται και παρακάτω

# In[55]:


f = open('fsa.txt' , 'w+')
cnt = 0
#print(lexicon[1600:1621])
for word in lexicon:
    if len(word)>1:
        f.write(format_arc(src=0,dst=cnt+1,src_sym=word[0],dst_sym=word[0],w=0))
        for j in range(cnt+2,len(word)+cnt+1):
            f.write(format_arc(src=j-1,dst=j,src_sym=word[j-cnt-1],dst_sym=word[j-cnt-1],w=0))    
        cnt = j+1
    #f.write(format_arc(src=cnt,dst=0,src_sym='<epsilon>',dst_sym='<epsilon>',w=0))
        f.write('{}\n'.format(cnt-1))
    else :
        f.write(format_arc(src=0,dst=cnt+1,src_sym=word[0],dst_sym=word[0],w=0))
        cnt+=1
        f.write('{}\n'.format(cnt))
f.close()
    


# In[56]:


get_ipython().run_cell_magic('bash', '', '\nfstcompile -isymbols=chars.syms -osymbols=chars.syms fsa.txt > fsa.fst;\nfstdeterminize fsa.fst > fsadet.fst \nfstminimize fsadet.fst > fsamin.fst\n#fstdraw --isymbols=chars.syms --osymbols=chars.syms fsamin.fst| dot -Tjpg >fsamin.jpg\nfstinfo fsa.fst | fstinfo fsamin.fst ')


# ## Βήμα 7ο
# 
# α)
# 
# Μέχρι στιγμής έχουμε δημιουργήσει έναν αποδοχέα για την κάθε λέξη που περιέχεται στο λεξικό μας και έναν μετατροπέα που υλοποιεί την Levenshtein απόσταση. Για την δημιουργία ενός ορθογράφου ελάχιστης απόστασης, ενός ορθογράφου ,δηλαδή,που μετατρέπει λέξεις εκτός λεξικού στην κοντινότερη (κατά Levenshtein) λέξη εντός του λεξικού μας, θα προβούμε στην σύνθεση του αποδοχέα με τον Levenshtein Transducer. Η παραπάνω σύνθεση πραγματοποιείται με την εντολή "fstcompose  Τ.fst fsamin.fst  comp.fst" , όπου το Τ.fst είναι ο Transducer μας και fsamin.fst είναι ο ελάχιστος απόδοχέας των λέξεων που περιέχονται στο λεξικό. Το αποτέλεσμα αυτής της σύνθεσης, καλείται min edit distance spell checker.

# In[57]:


get_ipython().run_cell_magic('bash', '', '\nfstcompose  Τ.fst fsamin.fst  comp.fst')


# ## Βήμα 8ο
# 
# α)
# 
# Για να δώσουμε κάποια λέξη σαν είσοδο στον ορθογράφο που μόλις συνθέσαμε, χρείαζεται η λέξη αυτή να βρίσκεται σε μορφή που μπορεί να αναγνωρίσει ο ορθογράφος μας. Για τον λόγο αυτό, προκείμένου να δώσουμε είσοδο στον min edit distance spell checker , δημιουργούμε έναν αποδοχέα για την λέξη την οποία θέλουμε να διορθώσουμε και στην συνέχεια συνθέτουμε τον αποδοχέα της υπό εξέταση λέξης με τον ορθογράφο μας. Η διορθωμένη λέξη(έξοδος του ορθογράφου) είναι η λέξη αυτή του λεξικού μας που απέχει λιγότερο από την υπό εξέταση λέξη (είσοδο του ορθογράφου). Συνεπώς, για να βρούμε την διορθωμένη λέξη , καλούμε την fstshortestpath στο αποτέλεσμα της σύνθεσης του ορθογράφου και του αποδοχέα της υπό εξέταση λέξης, η οποία βρίσκει το μονοπάτι με το χαμηλότερο βάρος.
# 
# β) 
# 
# Σε είσοδο cit ο ορθογράφος επιστρέφει την λέξη wit 
# 
# Δοκιμάζοντας,στην συνέχεια, τυχαία 20 λέξεις από το set που δίνεται μέσω του αντίστοιχου συνδέσμου, παρατηρούμε πως 14/20 λέξεις τις διόρθωσε σωστά ο ορθογράφος μας, γεγονός το οποίο αντιστοιχεί σε 70% επιτυχεία. Στο σημείο αυτό, αξίζει να παρατηρήσουμε ότι ο παραπάνω ορθογράφος παράγει ,τις πεισσότερες φορές, σωστά αποτελέσματα όταν η υπό εξέταση λέξη διαφέρει κατά ένα μόνο γράμμα από κάποια λέξη στο λεξικό μας. 

# In[1]:


letters = 'cit'
if letters not in lexicon:
    print('not found')
fi = open('input.txt','w+')
fi.write(format_arc(src=0,dst=1,src_sym=letters[0],dst_sym=letters[0],w=0))
for j in range(2,len(letters)+1):
    fi.write(format_arc(src=j-1,dst=j,src_sym=letters[j-1],dst_sym=letters[j-1],w=0))    
    #f.write(format_arc(src=cnt,dst=0,src_sym='<epsilon>',dst_sym='<epsilon>',w=0))
fi.write('{}\n'.format(j))
fi.close()   


# In[59]:


get_ipython().run_cell_magic('bash', '', '\nfstcompile -isymbols=chars.syms -osymbols=chars.syms input.txt > input.fst;\n\nfstcompose input.fst comp.fst test.fst \nfstshortestpath test.fst > test1.fst\nfsttopsort test1.fst test1.fst \n\nfstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait test1.fst| dot -Tpng >out.png\nfstprint --osymbols=chars.syms test1.fst | cut -f4 | grep -v "<epsilon>"|tr -d [:digit:]|tr -d \'\\n\'')


# ## Βήμα 9ο 
# 
# Στο σήμειο αυτό σταματάμε την ενασχόλησή μας με τις μηχανές πεπερασμένων καταστάσεων και στρεφόμαστε προς το word embeddings. Κάθε word embedding αποτελεί ,συνοπτικά, την αριθμητική αναπαράσταση μίας συγκεκριμένης λέξης προκειμένου αυτή να είναι διαχειρίσιμη από τον υπολογιστή. 
# 
# α) 
# 
# Αρχικά, επεξεργαζόμαστε ξανά το corpus που δημιουργήσαμε κατά το πρώτο βήμα της άσκησης. Αυτή την φορά, διαβάζουμε το αρχείο και στην συνέχεια δημιουργούμε μία λίστα από λίστες από συμβολοσειρές. Κάθε λίστα, μέσα στην γενική λίστα αντιστοιχεί σε μια πρόταση του corpus. Συγκεκριμένα, κάθε 'υπολίστα' περιέχει τα tokens της πρότασης στην οποία αντιστοιχεί.

# In[60]:


def tokenize_sentence(s):
    s = s.lower()
    s = s.replace('\n',' ')
    letters = string.ascii_lowercase + " " + "." + "?" + "!"  # characters from string.printable we don't want removed
    remove = ''.join([i for i in string.printable if i not in letters])     # characters that we want to remove
    remove = remove + '˜' + '£' + '¦' + '¨' + '©' + '½' + '“' + '”' + '€' + '™' + '' + '`' + '\'' + "‘" + "’"
    s = ''.join([i for i in s if i not in remove])     # keep rest of the text
    s = s.replace('.', '<stop>')
    s = s.replace('?', '<stop>')
    s = s.replace('!', '<stop>')
    sentences = s.split('<stop>')
    sentences_final = []
    for sentence in sentences:
        sen = sentence.split()
        sentences_final.append(sen)
    return sentences_final
f = open('corpus.txt', encoding="utf8")


# β) 
# 
# Στην συνέχεια, από την βιβλιοθήκη του gensim κάνουμε import το μοντέλο Word2Vec και με τις εντολές που φαίνονται παρακάτω επιτυγχάνουμε να δημιουργήσουμε word embeddings μεγέθους 100 με βάση τις προτάσεις που δημιουργήσαμε παραπάνω .

# In[61]:


from gensim.models import Word2Vec
import string
sentences = tokenize_sentence(corpus)

# Initialize word2vec. Context is taken as the 2 previous and 2 next words
model = Word2Vec(sentences, window=5, size=100, workers=12)
model.train(sentences, total_examples=len(sentences), epochs=1000)

# get ordered vocabulary list
voc = model.wv.index2word

# get vector size
dim = model.vector_size

# Convert to numpy 2d array (n_vocab x vector_size)
def to_embeddings_Matrix(model):  
    embedding_matrix = np.zeros((len(model.wv.vocab), model.vector_size))
    word2idx = {}
    for i in range(len(model.wv.vocab)):
        embedding_matrix[i] = model.wv[model.wv.index2word[i]]
        word2idx[model.wv.index2word[i]] = i
    return embedding_matrix, model.wv.index2word, word2idx


# γ)
# 
# Στο σημείο αυτό έχουμε δημιουργήσει τα Word embeddings για το λεξικό του corpus. Ένας τρόπος για να αξιολογήσουμε τα διανύσματα αυτά είναι , δεδομένης μίας λέξης του λεξιλογίου μας, να προσπαθήσουμε να βρούμε τις εννοιολογικά κοντινότερες σε αυτή λέξεις. Η εύρεση των 10 εννοιολογικά κοντινότερων λέξεων γίνεται με την εντολή model.wv.most_similar. Στο σημείο αυτό, σημείωνουμε ότι μετά την εύρεση των κοντινότερων λέξεων σε 10 τυχαίες λέξεις του λεξικού μας είδαμε πως για λέξεις όπως 'man', 'war' , 'god' το παραπάνω μοντέλο είναι σε θέση να βρεί λέξεις που όντως είναι συγγενείς. Αντίθετα, στην περίπτωση πιο αφηρημένων εννοιών, όπως liberty και madness, οι λέξεις που θεωρεί το μοντέλο ως εννοιολογικά συγγενείς στην πραγματικότητα είναι άσχετες μεταξύ τους.
# 
# Σε συνέχεια της προσπάθειάς μας για εύρεση των εννοιολογικα κοντινότερων λέξεων, εκπαιδεύουμε εκ νέου τα embeddings μας , αυτή την φορά επιλέγοντας μέγεθος παραθύρου ίσο με 10 διατηρώντας σταθερό τον αριθμό των εποχών. Τα αποτελέσματα τώρα που επιστρέφει το μοντέλο μας ως εννοιολογικά συγγενείς λέξεις βελτιώνονται σε κάποιο βαθμό. Αυτό συμβαίνει, διότι με την αύξηση του παραθύρου κατά την εκπαίδευση αυτόματα αυξάνεται και το μέγεθος του context κάθε λέξεις. Επομένως, με μεγαλύτερο context το μοντέλο "βλέπει" μια μεγαλύτερη γειτονία λέξεων γύρο από την υπό εξέταση λέξη, με αποτέλεσμα να μπορεί να προβλέπει καλύτερα τις κοντινότερες εννοιολογικά σε αυτή λέξεις.
# 
# Τέλος, επαναλαμβάνουμε τη δοκιμή μας με μεγαλύτερο αριθμό εποχών (1500). Όπως αναμέναμε, και εδώ τα αποτελέσματα είναι καλύτερα, δεδομένου ότι ο αυξημένος αριθμός εποχών συνεπάγεται πιο ακριβή υπολογισμό των βαρών και συνεπώς των πιθανοτήτων.

# In[62]:


input_=my_read("./words.txt")
words_tok=tokenize(input_)
for word in  words_tok:
    sim = model.wv.most_similar(word)
    print (word, ": ", sim, "\n")


# # Μέρος 1ο
# 
# Στο πρώτο αυτό μέρος της εργαστηριακής άσκησης, θα εστιάσουμε ξανά στην κατασκευή ενός spell checker. Αυτή την φορά ,ωστόσο, κατά την κατασκευή του transducer και του acceptor θα αποδώσουμε διαφορετικά βάρη στις ακμές τους.

# ## Βήμα 10ο 
# 
# Όπως αναφέραμε και παραπάνω τα βάρη ,αυτή την φορά, των ακμών στις μηχανές πεπερασμένης κατάστασης θα προκύπτουν με είτε βάση την συχνότητα εμφάνισης μιας λέξης στο κείμενο είτε με βάση την συχνότητα εμφάνισης κάθε χαρακτήρα στο κείμενο. Συνεπώς, για τον υπολογισμό των βαρών κρίνεται αναγκαίος ο υπολογισμός της πιθανότητας εμφάνισης της κάθε λέξης μέσα στο κείμενο, κάθως και την πιθανότητα εμφάνισης κάθε χαρακτήρα.
# 
# Αρχικά, δημιουργούμε 2 Python Dictionaries , τα οποίο ως key θα έχουν την λέξη ή τον χαρακτήρα αντίστοιχα και ως value θα έχουν την πιθανότητα εμφάνισης του αντίστοιχου key. 
# 
# Για την συμπλήρωση των 2 Dictionaries, διατρέχουμε την λίστα που δημιουργήσαμε κατά το βήμα 9α και για κάθε token που βρίσκουμε εκει εξετάζουμε αν υπάρχει στο Dictionary.Aν υπάρχει ήδη απλά αυξάνουμε το value του συγκεκριμένου, αλλίως θέτουμε την τιμή του  σε  μονάδα . Την ίδια ακριβώς διαδικασία ακολουθουμέ και για την κατασκευή του dictionary για τους χαρακτήρες του corpus μας.Με την παραπάνω διαδικάσια έχουμε καταφέρει να υπολογίσουμε και να αποθηκεύσουμε την συχνότητα εμφάνισης κάθε λέξης και κάθε χαρακτήρα στο Corpus.Συνεπώς, για τον υπολογισμό των αντίστοιχων πιθανοτήτων, διαιρούμε κάθε value του Word Dictionary με τον συνολικό αριθμό των λέξεων και κάθε value του Character Dictionary με τον συνολικό αριθμό των χαρακτήρων μέσα στο κείμενο. 

# In[63]:


word_dict = {}
symbol_dict = {}
temp =0
for word in tokens :
    if word not in word_dict:
        word_dict[word] = 0
    word_dict[word] += 1
for word in tokens :
    for symbol in word :
        if symbol not in symbol_dict:
            symbol_dict[symbol] = 0
        symbol_dict[symbol] += 1
total_chars = sum(symbol_dict.values())
total_words = sum(word_dict.values())

for key in word_dict:
    word_dict[key]= word_dict[key]/total_words

for key in symbol_dict:
    symbol_dict[key] = symbol_dict[key]/total_chars


# ## Βήμα 11ο 
# 
# Στο σημείο αυτό θα δημιουργήσουμε δύο Transducers που θα υλοποιούν την απόσταση Levenshtein. Η δύο αυτοί Transducers θα διαφοροποιούνται στο βάρος των edits. 
# 
# α)
# 
# Αρχικά, υπολογίζουμε το βάρος των edits του word level Transducer ως τον αρνητικό λογαριθμό της μέσης τιμής των πιθανοτήτων εμφάνισης κάθε λέξης του Corpus μας.
# 
# β) 
# 
# Στην συνέχεια, οπώς ακριβώς κάναμε στο βήμα 5 της προεργασίας κατασκευάζουμε τον word level Transducer μόνο που αυτή την φορά ορίζουμε ως βάρος για την εισαγωγή και την διαγραφή χαρακτήρα την τιμή που υπολογίσαμε στο ερώτημα α)

# In[64]:


temp = sum(word_dict.values())/len(word_dict)
w = -math.log(temp)

f = open('new_tran.txt', 'w+')
for i in symbols :
    if (i != '<epsilon>') :
        f.write(format_arc(src=0,dst=0,src_sym=i,dst_sym=i,w=0))
    for j in symbols :
        if (j != i ) :
            f.write(format_arc(src=0,dst=0,src_sym=i,dst_sym=j,w=w))
f.write('0\n')
f.close()


# γ)
# 
# Επαναλαμβάνουμε τις ενέργειες που πραγματοποιήσαμε κατά τα υποερωτήματα 11α, 11β. Έτσι, έχουμε ότι το βάρος των edits (εισαγωγή και διαγραφή κάποιου χαρακτήρα) του char level Transducer ισούται με τον αρνητικό λογάριθμο της μέσης τιμής της πιθανότητας εμφάνισης κάθε χαρακτήρα.
# 

# In[65]:


temp = 0 
temp = sum(symbol_dict.values())/len(symbol_dict)
wuni =-math.log(temp)

f = open('uni_tran.txt', 'w+')
for i in symbols :
    if (i != '<epsilon>') :
        f.write(format_arc(src=0,dst=0,src_sym=i,dst_sym=i,w=0))
    for j in symbols :
        if (j != i ) :
            f.write(format_arc(src=0,dst=0,src_sym=i,dst_sym=j,w=wuni))
f.write('0\n')
f.close()


# Για την ολοκλήρωση της κατασκεύης των 2 Transducer κάνουμε, όπως και παραπάνω, compile τα .txt αρχεία προκειμένου να δημιουργήσουμε τα αντίστοιχα .fst αρχεία τους.
# 
# Αυτή η μέθοδος απλά διασφαλίζει οτι οι τιμές των μετατροπών θα είναι συγκρίσιμες με τις τιμές στους αποδοχείς ώστε να μην δημιουργείται bias κατά "ακριβών" λέξεων αλλά ούτε να επιλέγεται πάντα η λέξη με τις λιγότερες μετατροπές.
# Ωστόσο συνεχίζουμε να δίνουμε σε όλες τις μετατροπές το ίδιο βάρος. Όπως αναφέραμε στην προεργασία αν είχαμε αρκετά μεγάλο corpus με λάθη θα μπορούσαμε να καταστήσουμε πιο φθηνές τις μετατροπές που αντιστοιχούν σε συχνότερα λάθη, έτσι ώστε να υπάρχει εξάρτηση του βάρους από τον τύπο της μετατροπής αλλά και από τα γράμματα που συμμετέχουν. Σε ένα ακόμα πιο σύνθετο μοντέλο τα βάρη θα μπορούσαν να εξαρώνται από το context του εκάτοτε γράμματος.

# In[66]:


get_ipython().run_cell_magic('bash', '', '\nfstcompile -isymbols=chars.syms -osymbols=chars.syms new_tran.txt > transducer.fst;\nfstcompile -isymbols=chars.syms -osymbols=chars.syms uni_tran.txt > uni_transducer.fst;')


# ## Βήμα 12ο 
# 
# α)
# 
# Σε πλήρη αναλογία με τις ενέργειες που πραγματοποιήσαμε στο βήμα 6 της προεργασίας , δημιουργούμε και εδώ 2 αποδεχείς για την κάθε λέξη του κειμένου μας. Ωστόσο, αυτή την φορά  δεν θεωρούμε πως ο word level αποδοχέας αποδέχεται την κάθε λέξη με μηδενικό βάρος. Αντίθετα, στον word level αποδοχέα , κάθε λέξη γίνεται δεκτή με βάρος ίσο με τον αρνητικό λογάριθμο της πιθανότητας εμφάνισής της. Κατα την κατασκεύη του συγκεκριμένου αποδοχέα, αποδίδουμε ολό το βάρος της λέξης στην πρώτο ακμή , ένω όλες οι υπόλοιπες ακμές (που δεν ξεκικάνε από την αρχική κατάσταση) έχουν πάλι βάρος μηδέν. 

# In[67]:



f = open('new_fsa.txt' , 'w+')
cnt = 0
#print(lexicon[1600:1621])
for word in lexicon:
    if len(word)>1:
        f.write(format_arc(src=0,dst=cnt+1,src_sym=word[0],dst_sym=word[0],w=-math.log(word_dict[word])))
        for j in range(cnt+2,len(word)+cnt+1):
            f.write(format_arc(src=j-1,dst=j,src_sym=word[j-cnt-1],dst_sym=word[j-cnt-1],w=0))    
        cnt = j+1
    #f.write(format_arc(src=cnt,dst=0,src_sym='<epsilon>',dst_sym='<epsilon>',w=0))
        f.write('{}\n'.format(cnt-1))
    else :
        f.write(format_arc(src=0,dst=cnt+1,src_sym=word[0],dst_sym=word[0],w=-math.log(word_dict[word])))
        cnt+=1
        f.write('{}\n'.format(cnt))
f.close()


# β) 
# 
# Στην συνέχεια, κάνουμε compile το .txt αρχείο που παράξαμε και έπειτα βελτιστοποιούμε τον αποδεχέα μας καλώντας τις εντολές fstrmepsilon , fstdeterminize, fstminize

# In[68]:


get_ipython().run_cell_magic('bash', '', '\nfstcompile -isymbols=chars.syms -osymbols=chars.syms new_fsa.txt > new_fsa.fst;\n\nfstdeterminize new_fsa.fst > new_fsadet.fst \nfstminimize new_fsadet.fst > new_fsamin.fst ')


# γ) 
# 
# Επαναλαμβάνουμε την ίδια διαδικάσια για την κατασκεύη και βελτιστοποίησει του char level αποδοχέα. Σε αύτη την περίπτωση ,ωστόσο, δεν αποδίδουμε ολοκληρό το βάρος της λέξης στην ακμή που ξεκινάει απο την αρχικά κατάσταση. Αντίθετα, το βάρος κάθε ακμής του εν λόγω αποδοχέα ισούται με τον αρνητικό λογάριθμο της πιθανότητας εμφάνισης του χαρακτήρα που πυροδοτεί την αντίστοιχη μετάβαση.

# In[69]:


f = open('uni_fsa.txt' , 'w+')
cnt = 0
#print(lexicon[1600:1621])
for word in lexicon:
    if len(word)>1:
        f.write(format_arc(src=0,dst=cnt+1,src_sym=word[0],dst_sym=word[0],w=-math.log(symbol_dict[word[0]])))
        for j in range(cnt+2,len(word)+cnt+1):
            f.write(format_arc(src=j-1,dst=j,src_sym=word[j-cnt-1],dst_sym=word[j-cnt-1],w=(-math.log(symbol_dict[word[j-cnt-1]]))/len(word)))    
        cnt = j+1
    #f.write(format_arc(src=cnt,dst=0,src_sym='<epsilon>',dst_sym='<epsilon>',w=0))
        f.write('{}\n'.format(cnt-1))
    else :
        f.write(format_arc(src=0,dst=cnt+1,src_sym=word[0],dst_sym=word[0],w=-math.log(symbol_dict[word[0]])))
        cnt+=1
        f.write('{}\n'.format(cnt))
f.close()


# In[70]:


get_ipython().run_cell_magic('bash', '', '\n\nfstcompile -isymbols=chars.syms -osymbols=chars.syms uni_fsa.txt > uni_fsa.fst \n\nfstdeterminize uni_fsa.fst > uni_fsadet.fst \nfstminimize uni_fsadet.fst > uni_fsamin.fst ')


# ## Βήμα 13ο 
# 
# α) 
# 
# Ακολουθώντας την ίδια διαδικασία με αυτή στο βήμα 7 της προεργασίας, συνθέτουμε τον word level Transducer με τον word level αποδοχέα κατασκευάζοντας έτσι τον νέο μας spell checker ο οποίος διορθώνει τις λέξεις λαμβάνοντας υπόψην και την πιθανότητα εμφάνισης κάθε λέξης μέσα στο λεξικό μας.

# In[72]:


get_ipython().run_cell_magic('bash', '', '\nfstarcsort --sort_type=olabel transducer.fst transducer.fst \nfstarcsort --sort_type=ilabel new_fsamin.fst new_fsamin.fst \n\nfstcompose transducer.fst new_fsamin.fst ncomp.fst ')


# β) 
# 
# Επαναλαμβάνουμε τα βήματα του υποερωτήματος α) και αυτή την φορά συνθέτουμε τον word level Transducer με τον unigram αποδοχέα

# In[73]:


get_ipython().run_cell_magic('bash', '', '\nfstarcsort --sort_type=ilabel uni_fsamin.fst uni_fsamin.fst \n\nfstcompose transducer.fst uni_fsamin.fst ucomp.fst')


# ### Σημέιωση : 
# Tα παρακάτω 2 cells αποτελούν υλοποίηση για την εισαγωγή λέξεων από τον χρήστη . Παρακάτω ακολουθεί υλοποίηση για αξιολόγηση των 2 spell checker εξετάζοντας όλες τις λέξεις του από το δοθέν αρχείο.

# In[74]:


letters = 'cit'
if letters not in lexicon:
    print('not found')
else:
    print(lexicon.index(letters))
fi = open('input1.txt','w+')
fi.write(format_arc(src=0,dst=1,src_sym=letters[0],dst_sym=letters[0],w=0))
for j in range(2,len(letters)+1):
    fi.write(format_arc(src=j-1,dst=j,src_sym=letters[j-1],dst_sym=letters[j-1],w=0))    
    #f.write(format_arc(src=cnt,dst=0,src_sym='<epsilon>',dst_sym='<epsilon>',w=0))
fi.write('{}\n'.format(j))
fi.close()   


# In[75]:


get_ipython().run_cell_magic('bash', '', '\nfstcompile -isymbols=chars.syms -osymbols=chars.syms input1.txt > word.fst;\nfstcompose word.fst ncomp.fst final.fst \nfstshortestpath final.fst > out1.fst \nfstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait out1.fst| dot -Tpng >out_WL.png\n\nfstcompose word.fst ucomp.fst ufinal.fst \nfstshortestpath ufinal.fst > out2.fst \nfsttopsort out2.fst out2.fst \nfsttopsort out1.fst out1.fst \nfstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait out2.fst| dot -Tpng >out_Uni.png\nfstprint --osymbols=chars.syms out2.fst | cut -f4 | grep -v "<epsilon>"|tr -d [:digit:]|tr -d \'\\n\'\necho \' \'\nfstprint --osymbols=chars.syms out1.fst | cut -f4 | grep -v "<epsilon>"|tr -d [:digit:]| tr -d \'\\n\'\n')


# *Eπεξήγηση : η πάνω  λέξη είναι η πρόβλεψη του unigram spell checker και η κάτω λέξη του word level
# 
# Tόσο ο word level spell checker όσο και ο unigram πρόβλεπουν λάθος την λέξη.
# 
# Για τον word level :
# 
# Με την fstcompose τα βάρη αθροίζονται (log semiring), δηλαδή προκύπτει η λέξη για την οποία το άθροισμα των μετατροπών και το -log(P(W)), γίνονται ελάχιστα. Όπως φαίνεται έχει επιλεγεί η λέξη it έχει μεγαλύτερη συχνότητα εμφάνισης από την sit. Έτσι αφού χρειαζόμαστε 1 μετατροπή για να φτάσουμε σε οποιαδήποτε από τις δύο επιλέγεται η πιο "φθηνή".
# Ισχύει γενικά Βάρος = -#insertions*log(P(ins)) - #deletions*log(P(dels)) - #substitions*log(P(subs)) - log(P(W))
# 
# Για τον unigram :
# Εδώ δεν πληρώνουμε την συχνότητα της λέξης αλλά το μήκος της. Καθώς όπως είπαμε τα βάρη δρουν αθροιστικά, έτσι για το sit θα έπρεπε να πληρώσουμε εκτός από το βάρος της μετατροπής του c σε s, το οποίο είναι ίδιο με τη διαγραφή του c, αλλά και το βάρος του s. Το πρόβλημα λοιπόν με το unigram μοντέλο είναι οτι πολύ συχνά οι μικρότερες λέξεις, όσο τους επιτρέπουν οι μετατροπές, αποκτούν προβάδισμα έναντι μεγαλύτερών τους.
# Ισχύει γενικά Βάρος = -#insertions*log(P(ins)) - #deletions*log(P(dels)) - #substitions*log(P(subs)) - log(P(w1) - log(P(w2)) - ... - log(P(wN))

# ## Βήμα 14ο 
# 
# α) 
# 
# Κατεβάζουμε το σύνολο των Evaluation δεδομένων που δίνονται  και τα αποθηκεύουμε σε ένα αρχείο test.txt
# 
# β) 
# 
# Στην συνέχεια διάβαζουμε το αρχείο με τα δεδομένα αξιολόγησης και με την βοήθεια της συνάρτησης που γράψαμε κατά το 2ο βήμα της προεργασίας λαμβάνουμε τα tokens σε κάθε γραμμή και τα αποθηκεύουμε σε μια λίστα . Το σύνολο των λιστών με τα tokens κάθε γραμμής τα αποθηκεύουμε όλα μάζι σε μια άλλη λίστα. Στην συνέχεια, δημιουργούμε έναν αποδοχέα για κάθε λάθος λέξη που υπάρχει μέσα στο αρχείο. Συνεχίζοντας, με την subprocess δημιουργούμε διεργασίες οι οποίες κάνουν compile όλους τους αποδοχείς των λέξεων και στην συνέχεια τους συνθέτουν αφενός με τον word level ορθογράφο που δημιουργήσαμε στο βήμα 13α και αφετέρου με τον ορθογράφο που δημιουργήσαμε στο 13β. Τέλος, για κάθε σύνθεση που έχει δημιουργηθεί βρίσκουμε το μονοπάτι με το μικρότερο μήκος και αυτό είναι η βέλτιστη πρόβλεψη των ορθογράφων. 

# In[76]:


f = open('test.txt','r+')
tests = f.readlines()
f.close()
cases=[]
cnt = 1
for x in tests:
    cases.append(tokenize(x))
for x in cases:
    for index in range(1,len(x)):
        word = x[index]
#         print(word)
        fi = open('input_{}.txt'.format(cnt),'w+')
        fi.write(format_arc(src=0,dst=1,src_sym=word[0],dst_sym=word[0],w=0))
        for j in range(2,len(word)+1):
            fi.write(format_arc(src=j-1,dst=j,src_sym=word[j-1],dst_sym=word[j-1],w=0))    
            #f.write(format_arc(src=cnt,dst=0,src_sym='<epsilon>',dst_sym='<epsilon>',w=0))
        fi.write('{}\n'.format(j))
        fi.close()   
        cnt+=1
subprocess.run('i=1; while [ $i -lt 271 ]; do fstcompile -isymbols=chars.syms -osymbols=chars.syms input_$i.txt > word_$i.fst; i=`expr $i + 1`;  done', shell=True)
subprocess.run('i=1; while [ $i -lt 271 ]; do fstcompose word_$i.fst ncomp.fst nfinal_$i.fst;echo $i; i=`expr $i + 1`;  done', shell=True)
subprocess.run('i=1; while [ $i -lt 271 ]; do fstcompose word_$i.fst ucomp.fst ufinal_$i.fst;echo $i; i=`expr $i + 1`;  done', shell=True)
f = open('noutput.txt','w+')
f.close()
f = open('uoutput.txt','w+')
f.close()
subprocess.run('i=1; while [ $i -lt 271 ]; do fstshortestpath nfinal_$i.fst >nout_$i.fst;fstrmepsilon nout_$i.fst nout_$i.fst;fsttopsort nout_$i.fst nout_$i.fst;fstprint -osymbols=chars.syms nout_$i.fst >> noutput.txt ;echo $i;i=`expr $i + 1`;  done', shell=True)
subprocess.run('i=1; while [ $i -lt 271 ]; do fstshortestpath ufinal_$i.fst >uout_$i.fst;fstrmepsilon uout_$i.fst uout_$i.fst;fsttopsort uout_$i.fst uout_$i.fst;fstprint -osymbols=chars.syms uout_$i.fst >> uoutput.txt ;echo $i;i=`expr $i + 1`;  done', shell=True)


# Με τις εντολές που ακολουθούν επεξεργαζόμαστε κατάλληλα το αρχείο .txt που μόλις δημιουργήσαμε , έτσι ώστε να κρατήσουμε μόνο τις λέξεις που προβλέπει ο κάθε ορθογράφος. Αφού απομονώσουμε τις προβλέψεις των ορθογράφων , τις αποθηκεύουμε σε μια λίστα ( ομοίως μια λίστα για κάθε ορθογράφο ). Στην συνέχεια, αποθηκεύουμε τις βέλτιστες λύσεις σε ένα αρχείο .txt ( ένα για κάθε ορθογράφο) . Τέλος, διατρέχουμε αυτή την λίστα και εξετάζουμε αν οι προβλέψεις αντιστοιχούν στις "σωστές" λέξεις , αθροίζουμε τις σωστές προβλέψεις και διαίρουμε με το πλήθος των λέξεων για να βρούμε το ποσοστό των σωστών προβλέψεων κάθε ορθογράφου. 

# In[77]:


get_ipython().run_cell_magic('bash', '', '\ngrep -v "<epsilon>" noutput.txt > nout1.txt\n#cut -f 4 nout1.txt > nout2.txt\ncat nout1.txt | tr  -d [:digit:]  > nout2.txt\ncat nout2.txt | tr  -d "." > nout3.txt\ncut -f 4 nout3.txt > nout4.txt\n#cat nout2.txt | tr -d "\\n" > nout3.txt\n\ngrep -v "<epsilon>" uoutput.txt > uout1.txt\n#cut -f 4 nout1.txt > nout2.txt\ncat uout1.txt | tr  -d [:digit:]  > uout2.txt\ncat uout2.txt | tr  -d "." > uout3.txt\ncut -f 4 uout3.txt > uout4.txt')


# In[79]:


f = open("nout4.txt","r")
s = f.readlines()
f.close()
nout= []
for element in s:
    nout.append(element[0])
f = open('nout_final.txt','w+')
for char in nout:
    f.write(char)
f.close()

f = open("uout4.txt","r")
s = f.readlines()
f.close()
uout= []
for element in s:
    uout.append(element[0])
f = open('uout_final.txt','w+')
for char in uout:
    f.write(char)

f = open('nout_final.txt','r')
npred = f.readlines()
ntemp =[]
nright = 0
cnt = 0
f.close()
for x in npred:
    ntemp+=(tokenize(x))
for i in cases:
#     print(i[0])
    for j in range(cnt,len(i)+cnt-1):
#         if j<=len(temp)-1:
            if ntemp[j]==i[0]:
                nright+=1
    cnt=j+1
print('The accuracy of word level spell checker is :',nright/(len(ntemp)-1))
f = open('uout_final.txt','r')
upred = f.readlines()
utemp =[]
uright = 0
cnt = 0

for x in upred:
    utemp+=(tokenize(x))
for i in cases:
#     print(i[0])
    for j in range(cnt,len(i)+cnt-1):
#         if j<=len(temp)-1:
            if utemp[j]==i[0]:
                uright+=1
    cnt=j+1
print('The accuracy of unigram spell checker is :',uright/(len(utemp)-1))


# γ) 
# 
# Μετά την ολοκλήρωση της αξιολόγησης των παραπάνω δύο ορθογράφων, προκύπτουν τα εξής αποτελέσματα : 
# Για τον word level spell checker έχουμε ποσοστό σωστών προβλέψεων 63.94% , ενώ για τον unigram spell checker έχουμε ποσοστό σωστών προβλέψεων ίσο με 58.36% . Από τα παραπάνω ποσοστό, παρατηρούμε πως ο word level ορθογράφος είναι καλύτερος συγκρητικά με τον αντίστοιχο unigram. Το παραπάνω αποτέλεσμα είναι, αναμμενόμενο κάθως στην περίπτωση του unigram ορθογράφου που η σώστη λέξη αποτελείται από σπάνιους χαρακτήρες , ο συγκεκριμένος ορθογράφος ,εξαιτίας του μεγάλου βάρους τον ακμών των edits που περιέχουν τους σπάνιους χαρακτήρες, θα προτιμήσει να διορθώσει την λέξη με χαρακτήρες που συνοδεύονται με ακμές χαμηλότερων βαρών και συνεπώς θα οδηγηθεί σε λάθος πρόβλεψει. Αντίθετα, ο word level ορθογράφος εξετάζει την πιθανότητα της λέξης ολόκληρης και συνεπώς για συνηθισμένες λέξεις θα έχει πολλή μεγαλή πιθανότητα να προβλέψει σωστά την λέξη.

#  # Μέρος 2
# 
# Στο δεύτερο μέρος της άσκησης αυτής θα εστίασουμε την προσοχή μας στην χρήση λεξικών αναπαραστάσεων για την δημιουργία ενός ταξινομητή συναισθήματος. Συγκεκριμένα, ως δεδομένα θα λάβουμε ορισμένες κριτικές για ταινίες από την σελίδα IMDB και προσπαθήσουμε να κατασκευάσουμε έναν ταξινομητή που θα μπορεί να ξεχωρίσει τα θετικά από τα αρνητικά σχόλια. 
# 

# ## Βήμα 16ο 
# 
# α) 
# 
# Αρχικά κατεβάζουμε τα δεδομένα που αναφέρονται στην εκφώνηση της άσκησης. Τα δεδομένα αυτά είναι  διαχωρισμένα σε  trainning set data  και test set data . Επιπλέον, στους επιμέρους φακέλους τα δεδομένα είναι επίσης διαχωρισμένα σε θετικά και αρνητικά σχόλια.
# 
# β)
# 
# Για την προεπεξεργασία των σχολίων ορίζουμε τις παρακάτω συναρτήσεις, μέσω των οποίων μπορούμε να πραγματοποιήσουμε το tokenization των σχολιών , αλλά και να δημιουργήσουμε ένα ενιαίο corpus που θα περιέχει ανακατεμένα τα σχολία.

# In[20]:


import os

data_dir = './aclImdb/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
pos_train_dir = os.path.join(train_dir, 'pos')
neg_train_dir = os.path.join(train_dir, 'neg')
pos_test_dir = os.path.join(test_dir, 'pos')
neg_test_dir = os.path.join(test_dir, 'neg')

# For memory limitations. These parameters fit in 8GB of RAM.
# If you have 16G of RAM you can experiment with the full dataset / W2V
MAX_NUM_SAMPLES = 12500
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = 1000000

import numpy as np

SEED = 42

# Fix numpy random seed for reproducibility
np.random.seed(42)

try:
    import glob2 as glob
except ImportError:
    import glob

import re

def strip_punctuation(s):
    return re.sub(r'[^a-zA-Z\s]', ' ', s)

def preprocess(s):
    return re.sub('\s+',' ', strip_punctuation(s).lower())

def tokenize(s):
    return s.split(' ')

def preproc_tok(s):
    return tokenize(preprocess(s))

def read_samples(folder, preprocess=lambda x: x):
    samples = glob.iglob(os.path.join(folder, '*.txt'))
    data = []
    for i, sample in enumerate(samples):
        if MAX_NUM_SAMPLES > 0 and i == MAX_NUM_SAMPLES:
            break
        with open(sample, 'r',encoding='latin-1') as fd:
            x = [preprocess(l) for l in fd][0]
            data.append(x)
    return data

def create_corpus(pos, neg):
    corpus = np.array(pos + neg)
    y = np.array([1 for _ in pos] + [0 for _ in neg])
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    return list(corpus[indices]), list(y[indices])


# Με τις παρακάτω εντολές έχουμε δημιουργήσει λίστες που περιέχουν ως string ολόκληρα τα σχόλια θετικά και αρνητικά , τόσο για το training set όσο και για το test set 

# In[21]:


pos_train = read_samples(pos_train_dir)
neg_train = read_samples(neg_train_dir)
pos_test = read_samples(pos_test_dir)
neg_test = read_samples(neg_test_dir,)
(corpus, y) = create_corpus(pos_train,neg_train)
(test_corp,test_y) = create_corpus(pos_test,neg_test)


# ## Βήμα 17ο 
# 
# α) 
# 
# TF-IDF Χρησιμοποιούμε BOW αναπαραστάσεις για να βρούμε τις φορές εμφάνισης κάθε λέξης σε μία κριτική. Αυτό αποτελεί μία ένδειξη της σημασίας της λέξης για το κείμενο. Ωστόσο η σημασία της λέξης δεν φαίνεται μόνο από τη συχνότητα της στο κείμενο αλλά και από την συχνότητα εμφανισής της σε άλλα κείμενα. Όσο σπανιότερα εμφανίζεται μία λέξη γενικά τόσο πιο σημαντικό το γεγονός της εμφάνισης της σε ένα κείμενο, καθώς αποτελεί μέσο διαχωρισμού του από τα υπόλοιπα. Αντίστροφα η εμφάνιση μίας λέξης που εμφανίζεται πάντα, όπως η λέξη "the" δεν προσφέρει κάτι το ιδιαίτερο στην κατηγοριοποίηση του κειμένου, οπότε δεν πρέπει να επηρεάζεται η επεξεργασία του από την εμφάνισή της.
# Για τον λόγο αυτό χρησιμοποιούνται τα βάρη TF-IDF. Ο term frequency tf(t, d) είναι ενδεικτικός των εμφανίσεων της λέξης στο κείμενο και συνήθως είναι η συχνότητα εμφάνισής της, δηλαδή ο αριθμός φορών εμφάνισής της δια τον συνολικό αριθμό λέξεων στο κείμενο, δηλαδή αντίστοιχο του BOW. Ο όρος inverse document frequency υλοποιεί αυτό που είπαμε παραπάνω, idf(t, D) ισούται με τον λογάριθμο του συνολικού αριθμού των κειμένων δια των αριθμό των κειμένων στα οποία εμφανίζεται η λέξη. Έτσι η λέξη "the", που λογικά εμφανίζεται σε κάθε κείμενο έχει idf("the", D) = 0 και αυτό σημαίνει οτι η εμφανισή της δεν προσφέρει καμία πληροφορία, tfidf("the", D) = tf("the", d)*idf("the", D) = 0. Αντίστροφα όσο λιγότερα τα κείμενα στα οποία εμφανίζεται μία λέξη τόσο μεγαλύτερη η σημασία της όταν εμφανίζεται, άρα τόσο μεγαλύτερος και ο όρος ifd("the", D).
# 
# β) 
# 
# Αρχικά, χρησιμοποιούμε τον CountVectorizer ο οποίος δημιουργεί one hot vector αναπαραστάσεις για την κάθε λέξη που περιέχεται στο corpus μας ,αποδίδοντας στην κάθε αναπαράσταση βάρος ανάλογο του της συχνότητας εμφάνισης αυτής της λέξης 

# In[22]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([('bow',CountVectorizer()),
                    ('classifier',LogisticRegression())])


# γ) 
# 
# Στην συνέχεια, με βάση το Bag of Words που δημιουργήθηκε από τον CountVectorizer εκπαιδεύουμε έναν logistic regression ταξινομήτη με βάση τα δεδομένα του training set. Στην συνέχεια, φορτώνουμε στον ταξινομητή τα δεδομένα από το test set και αποθηκεύουμε τις προβλέψεις του στην μεταβλητή predictions. Tέλος, συγκρίνουμε τα labels που προέβλεψε ο classifier με αυτά που θα έπρεπε να είχε βρει με την εντολή accuracy_score() και έτσι βρίσκουμε πως το ποσοστό των επιτυχημένων προβλέψεων του Logistic Regression ταξινομητή μας είναι 86.696%

# In[23]:



from sklearn.metrics import accuracy_score 

pipeline.fit(corpus,y)
predictions = pipeline.predict(test_corp)

test_y = np.asarray(test_y)
print(classification_report(predictions,test_y.flatten()))


# δ) 
# 
# Επαναλαμβάνουμε την ίδια διαδικασία μόνο που αυτή την φόρα αντι του CountVectorizer χρησιμοποιούμε τον TfidfVectorizer o οποίος στα one hot vector που παράγει αποδίδει tf-idf βάρη, όπως συνοπτικά περιγράψαμε στο α) ερώτημα. Στην συνεχεία με βάση τα νέα διανυσμάτα εκπαιδεύουμε τον ταξινομήτη μας και η ακρίβειά του αυτήτ την φορά ισούται με 88.28% . 
# 
# Η αύξηση αυτή της ακριβείας του ταξινομητή είναι αναμμενόμενη, κάθως όπως αναφέραμε και παραπάνω με την χρήση των  tf-idf βαρών γίνεται καλύτερη επιλογή features τα οποία θα επηρεάσουν την τελική απόφαση του εκτιμητή.

# In[24]:


from sklearn.feature_extraction.text import TfidfVectorizer
pipeline_tfidf = Pipeline([('bow',TfidfVectorizer()),
                    ('classifier',LogisticRegression())])
pipeline_tfidf.fit(corpus,y)
predictions_tfidf = pipeline_tfidf.predict(test_corp)

test_y = np.asarray(test_y)
print(classification_report(predictions_tfidf,test_y.flatten()))


# ## Βήμα 18 
# 
# α) 
# 
# Στο σημείο αυτό υπολογίζουμε το ποσοστό των λέξεων από το σύνολο των δεδομένων το οποίο δεν περιέχεται στην στις word embeddings αναπαραστάσεις του μοντέλου μας.

# In[25]:


l = []

for critic in corpus and test_corp:
    tokens = preproc_tok(critic)
    for word in tokens:
        l.append(word)

l = list(set(l))

oov = 0

for word in l:
    if word not in voc:
        oov+=1
rate = oov/len(l)
print(rate)


# β) 
# 
# Με βάση τις αναπαραστάσεις που έχει παράξει το Word2Vec μοντέλο μας υπολογίζουμε τις αναπαραστάσεις των σχολιών τόσο για τα trainig data όσο και για τα test data. Σημειώνεται πως οι ΟΟV λέξεις θεωρούμε πως αναπαρίστανται απο το μηδενικό διάνυσμα 

# In[26]:


def document_vector(word2vec_model, doc,voc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in voc]
    return np.mean(word2vec_model[doc], axis=0)


# In[27]:


import pandas as pd

corpus_train =  [0]*len(corpus)
for i in range (0,len(corpus)):
    corpus_train[i]=preproc_tok(corpus[i])
    
corpus_test = [0]*len(test_corp)
for i in range (0,len(test_corp)):
    corpus_test[i]=preproc_tok(test_corp[i])

imdb_train= []
for doc in corpus_train: #look up each doc in model
    imdb_train.append(document_vector(model, doc,voc))
df = pd.DataFrame(list(zip(y,imdb_train)),columns=["Y", "Comment"])

imdb_test = []
for doc in corpus_test: #look up each doc in model
    imdb_test.append(document_vector(model, doc,voc))
df2= pd.DataFrame(list(zip(test_y,imdb_test)),columns=["Y", "Comment"])


# In[28]:


logreg = LogisticRegression().fit(list(df["Comment"]), df["Y"])
predict = logreg.predict(list(df2["Comment"]))
print (classification_report(predict,df2["Y"]))


# γ)
# 
# Στην συνέχεια κατεβάζουμε τα ήδη εκπαιδεύμενα Embeddings Google Νews και επαναλαμβάνουμε το ερώτημα 9γ 

# In[31]:


from gensim.models import KeyedVectors
model_google = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',
binary=True, limit=NUM_W2V_TO_LOAD )
# [a,b,c] = to_embeddings_Matrix(model_google)
input_=my_read("./words.txt")
words_tok=preproc_tok(input_)
for word in  words_tok:
    sim = model_google.wv.most_similar(word)
    print (word, ": ", sim, "\n")


# Παρατηρούμε στο σημείο αυτό , πως οι λέξεις που προβλέπει τώρα το μοντέλο μας ως κοντινότερες εννοιολογικά ,είναι και στην πραγματικότητα εννοιολογικά κοντά. Το παραπάνω αποτέλεσμα είναι αναμμενόμενο, καθώς στην περίπτωση το pretrained Google Vectors το πλήθος των λέξεων είναι με διαφορά μεγαλύτερο από αυτό που προέκυπτε  στο δικό μας μοντέλο.

# ε) 
# 
# Στο σημείο αυτό δεδομένου, ότι αλλάξαμε Embeddings , πρέπει να υπολογίσουμε ξανά τις αναπαραστάσεις των σχολιών συμφώνα με τα νέα μας  Embeddings . Αφού, ολοκληρώσουμε την παραπάνω διαδικασία εκπαιδεύουμε εκ νέου τον ταξινομητή μας 

# In[32]:


def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.vocab]
    return np.mean(word2vec_model[doc], axis=0)

corpus_train =  [0]*len(corpus)

for i in range (0,len(corpus)):
    corpus_train[i]=preproc_tok(corpus[i])
    
corpus_test = [0]*len(test_corp)
for i in range (0,len(test_corp)):
    corpus_test[i]=preproc_tok(test_corp[i])

imdb_train= []
for doc in corpus_train: #look up each doc in model
    imdb_train.append(document_vector(model_google, doc))
df = pd.DataFrame(list(zip(y,imdb_train)),columns=["Y", "Comment"])

imdb_test = []
for doc in corpus_test: #look up each doc in model
    imdb_test.append(document_vector(model_google, doc))
df2= pd.DataFrame(list(zip(test_y,imdb_test)),columns=["Y", "Comment"])


# In[34]:


classif = LogisticRegression()
# Fit on data
clf = classif.fit(list(df["Comment"]), df["Y"])
predict = clf.predict(list(df2["Comment"]))
print (classification_report(predict,df2["Y"]))


# στ) Συνεχίζοντας, διατηρούμε ως Embeddings αυτά από το Google News. Ωστόσο, αυτή την φορά κατά την δημιουργία των αναπαραστάσεων των σχολιών θέλουμε να λάβουμε υπόψην μας και τα TF-IDF βάρη των λέξεων
# 

# In[46]:


def document_vectortfidf(word2vec_model, doc,weights):
    # remove out-of-vocabulary words
    weight =[]
    doc_new = []
    i=0
    for word in doc:
        if word in word2vec_model.vocab:            
            doc_new = doc_new+ [word]
            weight = weight + [weights[word]]
        i+=1
    tfidf=np.asarray(weight)
    return np.mean(word2vec_model[doc_new]*tfidf[:,np.newaxis], axis=0)


# In[47]:


vectorizer = TfidfVectorizer(analyzer = 'word',preprocessor = preprocess, tokenizer = preproc_tok)
X = vectorizer.fit_transform(corpus)


i=0
imdb_train= []
feature_names = vectorizer.get_feature_names()
for doc in corpus_train: #look up each doc in model
    feature_index = X[i,:].nonzero()[1]
    tfidf_scores={}
    for l in feature_index:
        tfidf_scores[feature_names[l]] =X[i, l]
    imdb_train.append(document_vectortfidf(model_google, doc,tfidf_scores))
    i+=1
df = pd.DataFrame(list(zip(y,imdb_train)),columns=["Y", "Comment"])


# In[49]:


X2 = vectorizer.fit_transform(test_corp)

i=0
imdb_test = []
feature_names2=vectorizer.get_feature_names()
for doc in corpus_test: #look up each doc in model
    feature_index = X2[i,:].nonzero()[1]
    tfidf_scores={}
    for l in feature_index:
        tfidf_scores[feature_names2[l]] =X2[i, l]
    #tfidf_scores = pd.DataFrame(list(zip([feature_names[l] for l in feature_index], [X[i, x] for x in feature_index])),columns = ["Feature_Index","Tf-Idf"])
    imdb_test.append(document_vectortfidf(model_google, doc,tfidf_scores))
    i+=1
df2= pd.DataFrame(list(zip(test_y,imdb_test)),columns=["Y", "Comment"])


# ζ) Ακολουθεί εκ νέου εκπαίδευση του ταξινομητή μας με τις νέες αναπαραστάσεις που δημιουργήσαμε στο προηγούμενο ερώτημα 

# In[50]:


classif = LogisticRegression()

# Fit on data
clf = classif.fit(list(df["Comment"]), df["Y"])
predict = clf.predict(list(df2["Comment"]))
print (classification_report(predict,df2["Y"]))


# Παρατηρούμε ότι με τη χρήση των βαρών, τα αποτελέσματα χειροτερεύουν . Αυτό μπορεί να εξηγηθεί, γιατί ο παράγοντας IDF απορρίπτει τις λέξεις που εμφανίζονται πολύ συχνά. Αυτό δεν σημαίνει ότι οι λέξεις αυτές δεν είναι χρήσιμες για την εξαγωγή συμπερασμάτων ως προς το αν ένα σχόλιο είναι θετικό ή αρνητικό.
