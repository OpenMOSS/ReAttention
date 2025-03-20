from mmengine.config import read_base

with read_base():
    
    # long task for long score
    
    ## longbench
    
    from ..datasets.longbench.longbenchnarrativeqa.longbench_narrativeqa_gen import LongBench_narrativeqa_datasets
    from ..datasets.longbench.longbenchqasper.longbench_qasper_gen import LongBench_qasper_datasets
    from ..datasets.longbench.longbenchmultifieldqa_en.longbench_multifieldqa_en_gen import LongBench_multifieldqa_en_datasets
    from ..datasets.longbench.longbenchmultifieldqa_zh.longbench_multifieldqa_zh_gen import LongBench_multifieldqa_zh_datasets

    from ..datasets.longbench.longbenchhotpotqa.longbench_hotpotqa_gen import LongBench_hotpotqa_datasets
    from ..datasets.longbench.longbench2wikimqa.longbench_2wikimqa_gen import LongBench_2wikimqa_datasets
    from ..datasets.longbench.longbenchmusique.longbench_musique_gen import LongBench_musique_datasets
    from ..datasets.longbench.longbenchdureader.longbench_dureader_gen import LongBench_dureader_datasets

    from ..datasets.longbench.longbenchgov_report.longbench_gov_report_gen import LongBench_gov_report_datasets
    from ..datasets.longbench.longbenchqmsum.longbench_qmsum_gen import LongBench_qmsum_datasets
    from ..datasets.longbench.longbenchmulti_news.longbench_multi_news_gen import LongBench_multi_news_datasets
    from ..datasets.longbench.longbenchvcsum.longbench_vcsum_gen import LongBench_vcsum_datasets

    from ..datasets.longbench.longbenchtrec.longbench_trec_gen import LongBench_trec_datasets
    from ..datasets.longbench.longbenchtriviaqa.longbench_triviaqa_gen import LongBench_triviaqa_datasets
    from ..datasets.longbench.longbenchsamsum.longbench_samsum_gen import LongBench_samsum_datasets
    from ..datasets.longbench.longbenchlsht.longbench_lsht_gen import LongBench_lsht_datasets

    from ..datasets.longbench.longbenchpassage_count.longbench_passage_count_gen import LongBench_passage_count_datasets
    from ..datasets.longbench.longbenchpassage_retrieval_en.longbench_passage_retrieval_en_gen import LongBench_passage_retrieval_en_datasets
    from ..datasets.longbench.longbenchpassage_retrieval_zh.longbench_passage_retrieval_zh_gen import LongBench_passage_retrieval_zh_datasets

    from ..datasets.longbench.longbenchlcc.longbench_lcc_gen import LongBench_lcc_datasets
    from ..datasets.longbench.longbenchrepobench.longbench_repobench_gen import LongBench_repobench_datasets

    ## leval
    
    # from ..datasets.leval.levaltpo.leval_tpo_gen import LEval_tpo_datasets
    # from ..datasets.leval.levalgsm100.leval_gsm100_gen import LEval_gsm100_datasets
    # from ..datasets.leval.levalquality.leval_quality_gen import LEval_quality_datasets
    # from ..datasets.leval.levalcoursera.leval_coursera_gen import LEval_coursera_datasets
    # from ..datasets.leval.levaltopicretrieval.leval_topic_retrieval_gen import LEval_tr_datasets
    # from ..datasets.leval.levalscientificqa.leval_scientificqa_gen import LEval_scientificqa_datasets
    
    # from ..datasets.leval.levalmultidocqa.leval_multidocqa_gen import LEval_multidocqa_datasets
    # from ..datasets.leval.levalpaperassistant.leval_paper_assistant_gen import LEval_ps_summ_datasets
    # from ..datasets.leval.levalnaturalquestion.leval_naturalquestion_gen import LEval_nq_datasets
    # from ..datasets.leval.levalfinancialqa.leval_financialqa_gen import LEval_financialqa_datasets
    # from ..datasets.leval.levallegalcontractqa.leval_legalcontractqa_gen import LEval_legalqa_datasets
    # from ..datasets.leval.levalnarrativeqa.leval_narrativeqa_gen import LEval_narrativeqa_datasets

    # from ..datasets.leval.levalnewssumm.leval_newssumm_gen import LEval_newssumm_datasets
    # from ..datasets.leval.levalgovreportsumm.leval_gov_report_summ_gen import LEval_govreport_summ_datasets
    # from ..datasets.leval.levalpatentsumm.leval_patent_summ_gen import LEval_patent_summ_datasets
    # from ..datasets.leval.levaltvshowsumm.leval_tvshow_summ_gen import LEval_tvshow_summ_datasets
    # from ..datasets.leval.levalmeetingsumm.leval_meetingsumm_gen import LEval_meetingsumm_datasets
    # from ..datasets.leval.levalreviewsumm.leval_review_summ_gen import LEval_review_summ_datasets

    ## infinitebench
    
    # # from ..datasets.infinitebench.infinitebenchcodedebug.infinitebench_codedebug_gen import InfiniteBench_codedebug_datasets
    # # from ..datasets.infinitebench.infinitebenchcoderun.infinitebench_coderun_gen import InfiniteBench_coderun_datasets
    # from ..datasets.infinitebench.infinitebenchendia.infinitebench_endia_gen import InfiniteBench_endia_datasets
    # from ..datasets.infinitebench.infinitebenchenmc.infinitebench_enmc_gen import InfiniteBench_enmc_datasets
    # from ..datasets.infinitebench.infinitebenchenqa.infinitebench_enqa_gen import InfiniteBench_enqa_datasets
    # from ..datasets.infinitebench.infinitebenchensum.infinitebench_ensum_gen import InfiniteBench_ensum_datasets
    # # # # from ..datasets.infinitebench.infinitebenchmathcalc.infinitebench_mathcalc_gen import InfiniteBench_mathcalc_datasets
    # # from ..datasets.infinitebench.infinitebenchmathfind.infinitebench_mathfind_gen import InfiniteBench_mathfind_datasets
    # from ..datasets.infinitebench.infinitebenchretrievekv.infinitebench_retrievekv_gen import InfiniteBench_retrievekv_datasets
    # from ..datasets.infinitebench.infinitebenchretrievenumber.infinitebench_retrievenumber_gen import InfiniteBench_retrievenumber_datasets
    # from ..datasets.infinitebench.infinitebenchretrievepasskey.infinitebench_retrievepasskey_gen import InfiniteBench_retrievepasskey_datasets
    # from ..datasets.infinitebench.infinitebenchzhqa.infinitebench_zhqa_gen import InfiniteBench_zhqa_datasets


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
