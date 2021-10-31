import argparse
def get_args():
    parser = argparse.ArgumentParser(description='Pytorch graph isomorphism network for graph classification')
    #these are parameters for attack model
    parser.add_argument('--effective', type=int, default=1)
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--search', type=int, default=1)
    #these are parameters for GIN model
    parser.add_argument('--dataset', type=str, default="IMDB-BINARY")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    dataset_name = args.dataset
    effective = int(args.effective)

    init_path = './out1/init_{}_{}_{}_{}_'.format(dataset_name,args.id, args.effective, args.search)
    with open(init_path+'search_type.txt', 'r') as f:
        search_type = eval(f.read())
    with open(init_path+'P.txt', 'r') as f:
        init_perturbation = eval(f.read())
    with open(init_path+'PR.txt', 'r') as f:
        init_perturbation_ratio = eval(f.read())
    with open(init_path+'D.txt', 'r') as f:
        init_distortion = eval(f.read())
    with open(init_path+'Q.txt', 'r') as f:
        init_query = eval(f.read())
    with open(init_path+'T.txt', 'r') as f:
        init_time = eval(f.read())
    
    our_path = './out1/our_{}_{}_{}_{}_'.format(dataset_name, args.id, args.effective, args.search)
    with open(our_path+'Q.txt', 'r') as f:
        our_query = eval(f.read())
    with open(our_path+'P.txt', 'r') as f:
        our_perturbation = eval(f.read())
    with open(our_path+'PR.txt', 'r') as f:
        our_perturbation_ratio = eval(f.read())
    with open(our_path+'D.txt', 'r') as f:
        our_distortion = eval(f.read())
    with open(our_path+'T.txt', 'r') as f:
        our_time = eval(f.read())
    
    random_path = './out1/ran_{}_{}_{}_{}_'.format(dataset_name, args.id, args.effective, args.search)
    with open(random_path+'P.txt', 'r') as f:
        ran_perturbation = eval(f.read())
    with open(random_path+'PR.txt', 'r') as f:
        ran_perturbation_ratio = eval(f.read())
    with open(random_path+'T.txt', 'r') as f:
        ran_time = eval(f.read())
    
    #所有的.txt中，攻击失败：值为-1， 无需攻击：值为0

    #delete no need instances
    #根据query挑选，query中只有0或>0，因为即便失败也会记录query次数
    L = len(our_query)
    index = []
    for i in range(L):
        if our_query[i] > 0:
            index.append(i)
    print(index)
    L = len(index) #number of target graphs
    print('the number of candadite test instances: {}'.format(L))
    
    search_type = [search_type[x] for x in index]
    init_distortion = [init_distortion[x] for x in index]
    init_perturbation = [init_perturbation[x] for x in index]
    init_perturbation_ratio = [init_perturbation_ratio[x] for x in index]
    init_query = [init_query[x] for x in index]
    init_time = [init_time[x] for x in index]
    our_distortion = [our_distortion[x] for x in index]
    our_perturbation = [our_perturbation[x] for x in index]
    our_perturbation_ratio = [our_perturbation_ratio[x] for x in index]
    our_query = [our_query[x] for x in index]
    our_time = [our_time[x] for x in index]
    ran_perturbation = [ran_perturbation[x] for x in index]
    ran_perturbation_ratio = [ran_perturbation_ratio[x] for x in index]
    ran_time = [ran_time[x] for x in index]
 
    #process query and time
    init_avg_query = sum(init_query) / L
    our_avg_query = sum(our_query) / L
    ran_avg_query = our_avg_query

    init_avg_time = sum(init_time) / L
    our_avg_time = sum(our_time) / L
    ran_avg_time = sum(ran_time) / L
    print('Init: avg query: {:.2f}'.format(init_avg_query))
    print('Init: avg attack time: {:.2f}'.format(init_avg_time))
    print('Our: avg query: {:.2f}'.format(our_avg_query))
    print('Our: avg attack time: {:.2f}'.format(our_avg_time))
    print('Ran: avg query: {:.2f}'.format(ran_avg_query))
    print('Ran: avg attack time: {:.2f}'.format(ran_avg_time))

    #process search type
    count_type = [0,0,0]
    for t in search_type:
        if t>=0:
            count_type[t] += 1
    print('the percentage of type1: {:.2f}'.format(count_type[0]/sum(count_type)*100))
    print('the percentage of type2: {:.2f}'.format(count_type[1]/sum(count_type)*100))
    print('the percentage of type3: {:.2f}'.format(count_type[2]/sum(count_type)*100))

    '''
    #process pertub of init
    init_distortion = [x for x in init_distortion if x>0]
    init_perturbation = [x for x in init_perturbation if x>0]
    init_perturbation_ratio = [x for x in init_perturbation_ratio if x>0]

    init_avg_distortion = sum(init_distortion) / L
    init_avg_perturbation = sum(init_perturbation) / L
    init_avg_pertub_ratio = sum(init_perturbation_ratio) / L * 100
    print('Init: avg distortion: {:.4f}'.format(init_avg_distortion))
    print('Init: avg perturbation: {:.4f}'.format(init_avg_perturbation))
    print('Init: avg perturb ratio: {:.4f}'.format(init_avg_pertub_ratio))
    '''

    #compute perturbation and distortion ubder different budget
    #先挑出来攻击成功的信息
    our_distortion = [x for x in our_distortion if x>0]
    our_perturbation = [x for x in our_perturbation if x>0]
    our_perturbation_ratio = [x for x in our_perturbation_ratio if x>0]
    ran_perturbation = [x for x in ran_perturbation if x>0]
    ran_perturbation_ratio = [x for x in ran_perturbation_ratio if x>0]
    init_distortion = [x for x in init_distortion if x>0]
    init_perturbation = [x for x in init_perturbation if x>0]
    init_perturbation_ratio = [x for x in init_perturbation_ratio if x>0]
    
    our_avg_perturbation = []
    our_avg_perturb_ratio = []
    our_avg_distortion = []
    init_avg_distortion = []
    init_avg_perturbation = []
    init_avg_perturb_ratio = []

    ran_avg_perturbation = []
    ran_avg_pertub_ratio = []

    our_success_ratio = []
    ran_success_ratio = []
    init_success_ratio = []
    for b in range(1, 21):
        budget = b / 100
        our_success_index = [x for x in list(range(len(our_distortion))) if our_perturbation_ratio[x] <= budget]
        our_success_distortion = [our_distortion[i] for i in our_success_index]
        our_success_perturbation = [our_perturbation[i] for i in our_success_index]
        our_success_perturb_ratio = [our_perturbation_ratio[i] for i in our_success_index]

        init_success_index = [x for x in list(range(len(init_distortion))) if init_perturbation_ratio[x]<= budget]
        init_success_distortion = [init_distortion[i] for i in init_success_index]
        init_success_perturbation = [init_perturbation[i] for i in init_success_index]
        init_success_perturb_ratio = [init_perturbation_ratio[i] for i in init_success_index]

        ran_success_index = [x for x in list(range(len(ran_perturbation))) if ran_perturbation_ratio[x] <= budget]
        ran_success_perturbation = [ran_perturbation[i] for i in ran_success_index]
        ran_success_perturb_ratio = [ran_perturbation_ratio[i] for i in ran_success_index]

        our_success_count = len(our_success_index)
        our_success_ratio.append(our_success_count / L)
        if our_success_count > 0:
            our_avg_perturbation.append(sum(our_success_perturbation) / our_success_count)
            our_avg_perturb_ratio.append(sum(our_success_perturb_ratio) / our_success_count)
            our_avg_distortion.append(sum(our_success_distortion) / our_success_count)
        else:
            our_avg_perturbation.append(0)
            our_avg_perturb_ratio.append(0)
            our_avg_distortion.append(0)
        
        init_success_count = len(init_success_index)
        init_success_ratio.append(init_success_count / L)
        if init_success_count > 0:
            init_avg_perturbation.append(sum(init_success_perturbation) / init_success_count)
            init_avg_perturb_ratio.append(sum(init_success_perturb_ratio) / init_success_count)
            init_avg_distortion.append(sum(init_success_distortion) / init_success_count)
        else:
            init_avg_perturbation.append(0)
            init_avg_perturb_ratio.append(0)
            init_avg_distortion.append(0)
        
        ran_success_count = len(ran_success_index)
        ran_success_ratio.append(ran_success_count / L)
        if ran_success_count > 0:
            ran_avg_perturbation.append(sum(ran_success_perturbation) / ran_success_count)
            ran_avg_pertub_ratio.append(sum(ran_success_perturb_ratio) / ran_success_count)
        else:
            ran_avg_perturbation.append(0)
            ran_avg_pertub_ratio.append(0)

    print('init success ratio'+'-'*20)
    for i in init_success_ratio:
        print('{:.4f}'.format(i))
    print('init avg perturbation'+'-'*20)
    for i in init_avg_perturbation:
        print('{:.4f}'.format(i))
    print('init avg perturb ratio'+'-'*20)
    for i in init_avg_perturb_ratio:
        print('{:.4f}'.format(i))
    print('init avg distortion' + '-'*20)
    for i in init_avg_distortion:
        print('{:.4f}'.format(i))

    print('our success ratio'+'-'*20)
    for i in our_success_ratio:
        print('{:.4f}'.format(i))
    print('our avg perturbation'+'-'*20)
    for i in our_avg_perturbation:
        print('{:.4f}'.format(i))
    print('our avg perturb ratio'+'-'*20)
    for i in our_avg_perturb_ratio:
        print('{:.4f}'.format(i))
    print('our avg distortion'+'-'*20)
    for i in our_avg_distortion:
        print('{:.4f}'.format(i))

    print('random success ratio'+'-'*20)
    for i in ran_success_ratio:
        print('{:.4f}'.format(i))
    print('random perturbation'+'-'*20)
    for i in ran_avg_perturbation:
        print('{:.4f}'.format(i))
    print('random perturb ratio'+'-'*20)
    for i in ran_avg_pertub_ratio:
        print('{:.4f}'.format(i))
    
    