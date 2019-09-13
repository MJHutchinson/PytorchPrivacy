Search.setIndex({docnames:["api","index","readme","source/pytorch_privacy.analysis","source/pytorch_privacy.analysis.moment_accountant","source/pytorch_privacy.analysis.moment_accountant.log_moment_utils","source/pytorch_privacy.analysis.moment_accountant.moment_accountant","source/pytorch_privacy.analysis.online_accountant","source/pytorch_privacy.analysis.pld_accountant","source/pytorch_privacy.analysis.pld_accountant.pld_accountant","source/pytorch_privacy.analysis.privacy_ledger","source/pytorch_privacy.analysis.utils","source/pytorch_privacy.dp_query","source/pytorch_privacy.dp_query.dp_query","source/pytorch_privacy.dp_query.gaussian_query","source/pytorch_privacy.optimizer","source/pytorch_privacy.optimizer.dp_optimizer","source/pytorch_privacy.optimizer.standard_optimizer","source/pytorch_privacy.optimizer.wrapper_optimizer","source/pytorch_privacy.utils","source/pytorch_privacy.utils.numpy_nest_utils","source/pytorch_privacy.utils.numpy_utils","source/pytorch_privacy.utils.test_numpy_utils","source/pytorch_privacy.utils.torch_nest_utils","source/pytorch_privacy.utils.torch_tensor_buffer"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api.rst","index.rst","readme.rst","source/pytorch_privacy.analysis.rst","source/pytorch_privacy.analysis.moment_accountant.rst","source/pytorch_privacy.analysis.moment_accountant.log_moment_utils.rst","source/pytorch_privacy.analysis.moment_accountant.moment_accountant.rst","source/pytorch_privacy.analysis.online_accountant.rst","source/pytorch_privacy.analysis.pld_accountant.rst","source/pytorch_privacy.analysis.pld_accountant.pld_accountant.rst","source/pytorch_privacy.analysis.privacy_ledger.rst","source/pytorch_privacy.analysis.utils.rst","source/pytorch_privacy.dp_query.rst","source/pytorch_privacy.dp_query.dp_query.rst","source/pytorch_privacy.dp_query.gaussian_query.rst","source/pytorch_privacy.optimizer.rst","source/pytorch_privacy.optimizer.dp_optimizer.rst","source/pytorch_privacy.optimizer.standard_optimizer.rst","source/pytorch_privacy.optimizer.wrapper_optimizer.rst","source/pytorch_privacy.utils.rst","source/pytorch_privacy.utils.numpy_nest_utils.rst","source/pytorch_privacy.utils.numpy_utils.rst","source/pytorch_privacy.utils.test_numpy_utils.rst","source/pytorch_privacy.utils.torch_nest_utils.rst","source/pytorch_privacy.utils.torch_tensor_buffer.rst"],objects:{"":{pytorch_privacy:[0,0,0,"-"]},"pytorch_privacy.analysis":{moment_accountant:[4,0,0,"-"],online_accountant:[7,0,0,"-"],pld_accountant:[8,0,0,"-"],privacy_ledger:[10,0,0,"-"],utils:[11,0,0,"-"]},"pytorch_privacy.analysis.moment_accountant":{log_moment_utils:[5,0,0,"-"],moment_accountant:[6,0,0,"-"]},"pytorch_privacy.analysis.moment_accountant.log_moment_utils":{generate_log_moments:[5,1,1,""],get_I1_I2_lambda:[5,2,1,""],integral_inf_mp:[5,2,1,""],pdf_gauss:[5,2,1,""],to_np_float_64:[5,2,1,""]},"pytorch_privacy.analysis.moment_accountant.moment_accountant":{compute_log_moments_from_ledger:[6,2,1,""],compute_online_privacy_from_ledger:[6,2,1,""],compute_privacy_loss_from_ledger:[6,2,1,""],get_privacy_spent:[6,2,1,""]},"pytorch_privacy.analysis.online_accountant":{OnlineAccountant:[7,3,1,""]},"pytorch_privacy.analysis.online_accountant.OnlineAccountant":{privacy_bound:[7,4,1,""],update_privacy:[7,4,1,""]},"pytorch_privacy.analysis.pld_accountant":{pld_accountant:[9,0,0,"-"]},"pytorch_privacy.analysis.pld_accountant.pld_accountant":{compute_online_privacy_from_ledger:[9,2,1,""],compute_privacy_loss_from_ledger:[9,2,1,""],get_FF1_add_remove:[9,1,1,""],get_FF1_substitution:[9,1,1,""],get_delta_add_remove:[9,2,1,""],get_delta_substitution:[9,2,1,""],get_eps_add_remove:[9,2,1,""],get_eps_add_remove_fixed_params:[9,2,1,""],get_eps_substitution:[9,2,1,""]},"pytorch_privacy.analysis.privacy_ledger":{GaussianSumQueryEntry:[10,3,1,""],PrivacyLedger:[10,3,1,""],QueryWithLedger:[10,3,1,""],QueryWithPerClientLedger:[10,3,1,""],SampleEntry:[10,3,1,""],format_ledger:[10,2,1,""]},"pytorch_privacy.analysis.privacy_ledger.GaussianSumQueryEntry":{l2_norm_bound:[10,4,1,""],noise_stddev:[10,4,1,""]},"pytorch_privacy.analysis.privacy_ledger.PrivacyLedger":{finalise_sample:[10,4,1,""],get_formatted_ledger:[10,4,1,""],record_sum_query:[10,4,1,""]},"pytorch_privacy.analysis.privacy_ledger.QueryWithLedger":{accumulate_preprocessed_record:[10,4,1,""],derive_sample_params:[10,4,1,""],get_noised_result:[10,4,1,""],get_record_derived_data:[10,4,1,""],initial_global_state:[10,4,1,""],initial_sample_state:[10,4,1,""],ledger:[10,4,1,""],merge_sample_states:[10,4,1,""],preprocess_record:[10,4,1,""],query:[10,4,1,""],set_ledger:[10,4,1,""]},"pytorch_privacy.analysis.privacy_ledger.QueryWithPerClientLedger":{accumulate_preprocessed_record:[10,4,1,""],derive_sample_params:[10,4,1,""],get_formatted_ledgers:[10,4,1,""],get_noised_result:[10,4,1,""],get_record_derived_data:[10,4,1,""],initial_global_state:[10,4,1,""],initial_sample_state:[10,4,1,""],ledgers:[10,4,1,""],merge_sample_states:[10,4,1,""],preprocess_record:[10,4,1,""],query:[10,4,1,""],set_ledgers:[10,4,1,""]},"pytorch_privacy.analysis.privacy_ledger.SampleEntry":{population_size:[10,4,1,""],queries:[10,4,1,""],selection_probability:[10,4,1,""]},"pytorch_privacy.analysis.utils":{grab_pickled_accountant_results:[11,2,1,""],set_accountant_tables_dir:[11,2,1,""]},"pytorch_privacy.dp_query":{dp_query:[13,0,0,"-"],gaussian_query:[14,0,0,"-"]},"pytorch_privacy.dp_query.dp_query":{DPQuery:[13,3,1,""],SumAggregationDPQuery:[13,3,1,""]},"pytorch_privacy.dp_query.dp_query.DPQuery":{accumulate_preprocessed_record:[13,4,1,""],accumulate_record:[13,4,1,""],derive_sample_params:[13,4,1,""],get_noised_result:[13,4,1,""],initial_global_state:[13,4,1,""],initial_sample_state:[13,4,1,""],merge_sample_states:[13,4,1,""],preprocess_record:[13,4,1,""],set_ledger:[13,4,1,""]},"pytorch_privacy.dp_query.dp_query.SumAggregationDPQuery":{accumulate_preprocessed_record:[13,4,1,""],initial_sample_state:[13,4,1,""],merge_sample_states:[13,4,1,""]},"pytorch_privacy.dp_query.gaussian_query":{GaussianDPQuery:[14,3,1,""],NumpyGaussianDPQuery:[14,3,1,""]},"pytorch_privacy.dp_query.gaussian_query.GaussianDPQuery":{derive_sample_params:[14,4,1,""],get_noised_result:[14,4,1,""],get_record_derived_data:[14,4,1,""],initial_global_state:[14,4,1,""],initial_sample_state:[14,4,1,""],make_global_state:[14,4,1,""],preprocess_record:[14,4,1,""],set_ledger:[14,4,1,""]},"pytorch_privacy.dp_query.gaussian_query.NumpyGaussianDPQuery":{accumulate_preprocessed_record:[14,4,1,""],derive_sample_params:[14,4,1,""],get_noised_result:[14,4,1,""],get_record_derived_data:[14,4,1,""],initial_global_state:[14,4,1,""],initial_sample_state:[14,4,1,""],make_global_state:[14,4,1,""],merge_sample_states:[14,4,1,""],preprocess_record:[14,4,1,""],set_ledger:[14,4,1,""]},"pytorch_privacy.optimizer":{dp_optimizer:[16,0,0,"-"],standard_optimizer:[17,0,0,"-"],wrapper_optimizer:[18,0,0,"-"]},"pytorch_privacy.optimizer.dp_optimizer":{DPGaussianOptimizer:[16,3,1,""],DPOptimizer:[16,3,1,""]},"pytorch_privacy.optimizer.dp_optimizer.DPGaussianOptimizer":{ledger:[16,4,1,""]},"pytorch_privacy.optimizer.dp_optimizer.DPOptimizer":{apply_grads:[16,4,1,""],fit_batch:[16,4,1,""],get_grads:[16,4,1,""],get_logged_statistics:[16,4,1,""],get_step_summary:[16,4,1,""]},"pytorch_privacy.optimizer.standard_optimizer":{StandardOptimizer:[17,3,1,""]},"pytorch_privacy.optimizer.standard_optimizer.StandardOptimizer":{fit_batch:[17,4,1,""],get_logged_statistics:[17,4,1,""],get_step_summary:[17,4,1,""]},"pytorch_privacy.optimizer.wrapper_optimizer":{WrapperOptimizer:[18,3,1,""]},"pytorch_privacy.optimizer.wrapper_optimizer.WrapperOptimizer":{fit_batch:[18,4,1,""],get_logged_statistics:[18,4,1,""],get_step_summary:[18,4,1,""]},"pytorch_privacy.utils":{numpy_nest_utils:[20,0,0,"-"],numpy_utils:[21,0,0,"-"],test_numpy_utils:[22,0,0,"-"],torch_nest_utils:[23,0,0,"-"],torch_tensor_buffer:[24,0,0,"-"]},"pytorch_privacy.utils.numpy_nest_utils":{apply_to_structure:[20,2,1,""],flatten:[20,2,1,""],map_structure:[20,2,1,""],reduce_structure:[20,2,1,""],structured_lists_to_lists:[20,2,1,""],structured_ndarrays_to_lists:[20,2,1,""]},"pytorch_privacy.utils.numpy_utils":{add_parameters:[21,2,1,""],clip:[21,2,1,""],gaussian_noise:[21,2,1,""],subtract_params:[21,2,1,""],to_pure_python:[21,2,1,""]},"pytorch_privacy.utils.test_numpy_utils":{test_clip:[22,2,1,""]},"pytorch_privacy.utils.torch_nest_utils":{flatten:[23,2,1,""],map_structure:[23,2,1,""],parameters_to_tensor_groups:[23,2,1,""],reduce_structure:[23,2,1,""]},"pytorch_privacy.utils.torch_tensor_buffer":{TensorBuffer:[24,3,1,""]},"pytorch_privacy.utils.torch_tensor_buffer.TensorBuffer":{append:[24,4,1,""],capacity:[24,4,1,""],current_size:[24,4,1,""],values:[24,4,1,""]},pytorch_privacy:{analysis:[3,0,0,"-"],dp_query:[12,0,0,"-"],optimizer:[15,0,0,"-"],utils:[19,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","attribute","Python attribute"],"2":["py","function","Python function"],"3":["py","class","Python class"],"4":["py","method","Python method"]},objtypes:{"0":"py:module","1":"py:attribute","2":"py:function","3":"py:class","4":"py:method"},terms:{"abstract":[13,18],"class":[7,10,13,14,16,17,18,24],"function":[13,23],"new":[7,9,13,14,24],"return":[6,7,9,10,13,14,16,17,18,20,21,23],"true":9,For:[9,20],NOT:[6,9],The:[6,7,9,10,13,14,16,20,23,24],abc:[13,18],abl:[16,17,18],about:[10,13,16,17,18],abs:[6,9,13],account:[6,7,9,10],accountancy_paramet:7,accountancy_update_method:7,accumul:[13,14,20,23],accumulate_preprocessed_record:[10,13,14],accumulate_record:13,adapt:24,add:21,add_paramet:21,add_remov:9,addit:[7,9,21],adjac:9,adjacency_definit:9,after:[13,14],aggreg:[13,14],aim:[6,9],alia:10,all:[6,10,13,14,20],allow:[17,18],also:14,analysi:[0,1],ani:[16,17,18],append:24,appli:[9,10,13,14,20,23],applic:1,apply_grad:16,apply_to_structur:20,approxim:9,arg:16,arrai:[9,10],arxiv:[6,9,13],assum:[7,9,14,21],attribut:23,base:[7,10,13,14,16,17,18,24],batch:[17,18],been:[10,13,14],between:20,blank:13,bound:[6,7,9,13,21],buffer:24,can:9,capac:24,clean:18,client:13,clip:[9,10,13,14,16,17,18,21],code:[9,17],com:9,compar:17,compos:[6,10],comput:[6,9],compute_log_moments_from_ledg:6,compute_online_privacy_from_ledg:[6,9],compute_privacy_loss_from_ledg:[6,9],concaten:20,consid:[6,9],contain:[13,14,21],convert:10,correct:10,correspond:21,could:[10,13,16,17,18],current:[7,13,14],current_s:24,data:[9,10,13,23],dataset:[9,10],deduc:14,defin:13,definit:[9,13],deleg:13,delta:[6,9],deriv:[13,14],derive_sample_param:[10,13,14],design:[16,17,18],desir:[16,17,18,23],detail:10,deviat:10,dict:[20,21],dictionari:[14,16,17,18,20],differ:7,differenti:[2,16,17,18],discretis:9,distribut:9,doe:[13,14,17],doing:[16,17,18],done:10,dp_optim:[0,1,15],dp_queri:[0,1,10],dp_sum_queri:16,dpbay:9,dpgaussianoptim:16,dpoptim:16,dpqueri:[10,13,14],dqqueri:10,dtype:24,each:[21,23],easi:17,effect:9,effective_z_t:9,either:18,element:[20,21,23],end:24,ensur:10,enter:[6,9],entri:[6,7,10,20],epsilon:[6,9],event:[10,13,14],everi:20,exact:9,execut:10,exist:18,expect:23,experi:7,extend:18,extra:13,extract:23,extrect:23,f_prod:9,far:[13,14],fashion:[9,13],fft:9,field:10,filenam:11,finalis:10,finalise_sampl:10,first:[13,14,21,23],fit:[17,18],fit_batch:[16,17,18],fix:[6,9,13],flatten:[20,23],flatter:10,float32:24,fly:[16,17,18],format:[7,10],format_ledg:10,from:[7,9,13,21,23],gaussian:[6,9,14,16],gaussian_nois:21,gaussian_queri:[0,1,12],gaussiandpqueri:14,gaussiansumqueryentri:10,gener:13,generate_log_mo:5,get_delta_add_remov:9,get_delta_substitut:9,get_eps_add_remov:9,get_eps_add_remove_fixed_param:9,get_eps_substitut:9,get_ff1_add_remov:9,get_ff1_substitut:9,get_formatted_ledg:10,get_grad:16,get_i1_i2_lambda:5,get_logged_statist:[16,17,18],get_noised_result:[10,13,14],get_privacy_sp:6,get_record_derived_data:[10,14],get_step_summari:[16,17,18],github:9,given:[6,9,10,13,14,20,24],global:[13,14],global_paramet:[13,14],global_st:[10,13,14],grab_pickled_accountant_result:11,grad:[10,16,23],gradient:[13,14,16,17,18],group:[10,13,14,23],have:[7,10,13,14],hold:24,http:[6,9,13],ident:[18,20,23],implement:[2,14],increment:[6,9],incremented_ledg:7,index:1,individu:10,inform:[13,16,17,18],initi:[13,14,20],initial_global_st:[10,13,14],initial_sample_st:[10,13,14],input:[14,16,23],inspect:[16,17,18],integral_inf_mp:5,interest:[16,17,18],interfac:[1,14,18],issu:10,just:[10,13,14,23],keep:10,kei:[20,21],kwarg:16,l2_clipping_bound:10,l2_norm_bound:10,l2_norm_clip:[14,16],lambda_v:5,ledger:[6,7,9,10,13,14,16],length:9,lightweight:24,list:[9,10,21,23],log:[6,16,17,18],log_moment:6,log_moment_util:[0,1,3,4],loss:[6,9,16],loss_per_exampl:[16,17],make_global_st:14,manner:[7,18],map_structur:[20,23],max:10,max_lambda:6,maximum:6,mean:[5,21],mechan:[6,9,10,16],merg:[13,14],merge_sample_st:[10,13,14],method:[7,18,24],metric:[16,17,18],microbatch:10,might:[16,17,18],minbatch:10,mircobatch:13,model:[16,17],modul:[0,1,3,4,8,12,15,19],moment:6,moment_account:[0,1,3],monitor:[16,17,18],name:10,ncomp:9,neighbour:9,nest:[16,17,18,20],new_dir:11,new_global_st:[13,14],next:[13,14],nois:[9,10,13,14,16],noise_multipli:16,noise_stddev:[10,14],non:17,none:[6,7,9,10,16],norm:[10,13,14],num_client:10,num_microbatch:16,number:[9,10],numpi:[9,10,14],numpy_nest_util:[0,1,19],numpy_util:[0,1,19],numpygaussiandpqueri:14,object:[7,10,24],occur:[10,14],ocnt:23,one:[6,9],onli:7,onlin:[7,9],online_account:[0,1,3],onlineaccount:7,oper:[2,10,20,23],optim:[0,1,23],optimis:[16,17,18],org:[6,9,13],origin:6,other:13,out:23,output:16,packag:[0,1],page:1,pair:[10,17,18,23],paper:[6,9],param:[6,9,10,13,14,16,17,18,20,21,23],param_dict:20,param_group:[13,14,16,23],param_groups_1:23,param_groups_2:23,paramet:[6,7,9,10,13,14,21,23,24],parameter_group:23,parameters_to_tensor_group:23,parent_kei:20,pass:18,pdf1:5,pdf2:5,pdf_gauss:5,per:9,perform:[2,7,16],place:20,pld:9,pld_account:[0,1,3],point:9,population_s:10,possibl:18,prepend:20,preprocess:13,preprocess_record:[10,13,14],preprocessed_record:10,previou:[6,9],previous:[6,7],privaci:[2,6,7,9,10,13,14,16,17,18],privacy_bound:7,privacy_ledg:[0,1,3],privacyledg:[10,13,14],privat:16,probabl:9,process:[13,16,17,18],program:1,properti:[7,9,10,16,24],purpos:[16,17,18],pytorch:[2,16,18],pytorch_privaci:[0,1],pytorchprivaci:1,q_t:9,queri:[6,9,10,13,14],query_arrai:10,querywithledg:10,querywithperclientledg:10,rang:9,recodr:[13,14],record:[10,13,14],record_sum_queri:10,reduc:[20,23],reduce_structur:[20,23],referenc:9,relev:[16,17,18],remov:9,report:[16,17,18],repres:23,represent:10,request:23,requir:7,rest:20,result:[10,13,14,23],run:10,same:[13,14,20,21,23],sampl:[9,10,13,14],sample_arrai:10,sample_st:[10,13,14],sample_state_1:[10,13,14],sample_state_2:[10,13,14],sampleentri:10,scale:14,search:1,second:[13,14,23],see:10,seen:[6,7],select:9,selected_indic:10,selection_prob:10,sep:20,separ:20,seri:[20,24],set:[6,9,10,13,14,20,23],set_accountant_tables_dir:11,set_ledg:[10,13,14],sgd:[13,14],shape:[13,14,20,21,23,24],should:7,sigma:[5,9],simpl:[10,13,14,16,17,18],simplic:10,singl:[9,13,14,16,17,18,20,23],size:24,some:[13,14],sourc:[5,6,7,9,10,11,13,14,16,17,18,20,21,22,23,24],specif:[16,23],specifi:9,speed:[7,9],standard:[10,18],standard_optim:[0,1,15],standardoptim:17,start:13,stata:[13,14],state:[13,14,23],statist:[16,17,18],std:21,step:[16,17,18],store:10,structur:[13,16,17,18,20,23],structured_lists_to_list:20,structured_ndarrays_to_list:20,submodul:[0,1],subpackag:[0,1],substitut:9,subtract:21,subtract_param:21,sum:[10,13,14],sumaggregationdpqueri:[13,14],summari:[6,16,17,18],summaris:6,summat:13,target:[6,9],target_delta:[6,9],target_ep:[6,9],templat:[10,13],tensor:[10,13,16,17,21,24],tensor_group:23,tensorbuff:[10,24],terminolog:13,test_clip:22,test_numpy_util:[0,1,19],thi:[6,7,10,13,14,16,17,18,23],think:10,those:9,through:10,to_np_float_64:5,to_pure_python:21,togeth:[10,13,14,21],torch:[16,17,24],torch_nest_util:[0,1,19],torch_tensor_buff:[0,1,19],total:[6,16],total_log_mo:6,truncat:9,tupl:[10,13,14],two:[13,14,23],type:7,under:9,updat:[7,13,14],update_privaci:7,upper:6,use:[9,10,13,14],used:9,using:[6,7,9,17,18],usual:[13,14],util:[0,1,2,3,17],utilis:[6,9,24],val:21,valu:24,variou:2,vector:13,version:10,via:[13,14,23,24],wai:10,were:10,what:10,when:13,where:14,whether:14,which:23,whole:[6,9,10],worri:10,wrap:[17,18],wrapper:[10,16,17],wrapper_optim:[0,1,15,16,17],wrapperoptim:[16,17,18],you:10,zero:[13,14]},titles:["Application Programming Interface","Welcome to Pytorch Privacy\u2019s documentation!","PytorchPrivacy","pytorch_privacy.analysis package","pytorch_privacy.analysis.moment_accountant package","pytorch_privacy.analysis.moment_accountant.log_moment_utils module","pytorch_privacy.analysis.moment_accountant.moment_accountant module","pytorch_privacy.analysis.online_accountant module","pytorch_privacy.analysis.pld_accountant package","pytorch_privacy.analysis.pld_accountant.pld_accountant module","pytorch_privacy.analysis.privacy_ledger module","pytorch_privacy.analysis.utils module","pytorch_privacy.dp_query package","pytorch_privacy.dp_query.dp_query module","pytorch_privacy.dp_query.gaussian_query module","pytorch_privacy.optimizer package","pytorch_privacy.optimizer.dp_optimizer module","pytorch_privacy.optimizer.standard_optimizer module","pytorch_privacy.optimizer.wrapper_optimizer module","pytorch_privacy.utils package","pytorch_privacy.utils.numpy_nest_utils module","pytorch_privacy.utils.numpy_utils module","pytorch_privacy.utils.test_numpy_utils module","pytorch_privacy.utils.torch_nest_utils module","pytorch_privacy.utils.torch_tensor_buffer module"],titleterms:{analysi:[3,4,5,6,7,8,9,10,11],applic:0,document:1,dp_optim:16,dp_queri:[12,13,14],gaussian_queri:14,indic:1,interfac:0,log_moment_util:5,modul:[5,6,7,9,10,11,13,14,16,17,18,20,21,22,23,24],moment_account:[4,5,6],numpy_nest_util:20,numpy_util:21,online_account:7,optim:[15,16,17,18],packag:[3,4,8,12,15,19],pld_account:[8,9],privaci:1,privacy_ledg:10,program:0,pytorch:1,pytorch_privaci:[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],pytorchprivaci:2,standard_optim:17,submodul:[3,4,8,12,15,19],subpackag:3,tabl:1,test_numpy_util:22,torch_nest_util:23,torch_tensor_buff:24,util:[11,19,20,21,22,23,24],welcom:1,wrapper_optim:18}})