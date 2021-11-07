import json
import os
import argparse

def exportNCFPlease(transform_dim,
                    lr,
                    batch_size,
                    dataset_path,
                    save_path="../configs/"):

    ## generate conf.json
    ncf_conf={
    "dsl_version": 2,
    "initiator": {
    "role": "guest",
    "party_id": 9999
  },
  "role": {
    "arbiter": [10000],
    "host": [10000, 9999],
    "guest": [9999]
  },
  "job_parameters": {
    "common": {
      "work_mode": 0,
      "backend": 0
    }
  },
  "component_parameters": {
    "common": {
      "homo_nn_0": {
        "api_version": 2,
        "encode_label": True,
        "max_iter": 2,
        "batch_size": batch_size,
        "early_stop": {
          "early_stop": "diff",
          "eps": 0.0001
        },
        "optimizer": {
          "lr": lr,
          "optimizer": "Adam"
        },
        "loss": "NLLLoss",
        "metrics": ["accuracy"],
        "nn_define": [
          {
            "layer": "Linear",
            "in_features": transform_dim,
              "out_features": transform_dim
          },

          {
            "layer": "ReLU"
          },
          {
            "layer": "Linear",
            "in_features": transform_dim,
            "out_features": transform_dim
          },
          {
            "layer": "ReLU"
          },
          {
            "layer": "Linear",
            "in_features": transform_dim,
            "out_features": transform_dim
          }
        ],
        "config_type": "pytorch"
      }
    },
    "role": {
      "host": {
        "0": {
          "reader_0": {
            "table": {
              "name": "recommend_matrix",
              "namespace": "experiment"
            }
          }
        },
        "1": {
          "reader_0": {
            "table": {
              "name": "recommend_matrix",
              "namespace": "experiment"
            }
          }
        }
      },
      "guest": {
        "0": {
          "reader_0": {
            "table": {
              "name": "recommend_matrix",
              "namespace": "experiment"
            }
          }
        }
      }
    }
  }
}
    ## generate dsl.json
    dsl_conf={
  "components": {
    "reader_0": {
        "module": "Reader",
        "output": {
            "data": [
                "data"
                ]
        }
    },
      "homo_nn_0": {
      "module": "ncf",
      "input": {
        "data": {
          "train_data": [
            "reader_0.data"
          ]
        }
      },
      "output": {
        "data": [
          "data"
        ],
        "model": [
          "model"
        ]
      }
    }
  }
}

    ## generate test_suite.json
    suite_dict={
  "data": [
    {
      "file": dataset_path,
      "id_delimiter": ",",
      "head": 1,
      "partition": 4,
      "work_mode": 0,
      "backend": 0,
      "storage_engine": "PATH",
      "namespace": "experiment",
      "table_name": "recommend_matrix",
      "use_local_data": 0,
      "role": "guest_0"
    },
    {
      "file": dataset_path,
      "id_delimiter": ",",
      "head": 1,
      "partition": 4,
      "work_mode": 0,
      "backend": 0,
      "storage_engine": "PATH",
      "namespace": "experiment",
      "table_name": "recommend_matrix",
      "use_local_data": 0,
      "role": "host_0"
    },
    {
      "file": dataset_path,
      "id_delimiter": ",",
      "head": 1,
      "partition": 4,
      "work_mode": 0,
      "backend": 0,
      "storage_engine": "PATH",
      "namespace": "experiment",
      "table_name": "recommend_matrix",
      "use_local_data": 0,
      "role": "host_1"
    }
  ],
  "tasks": {
    "recommend_update": {
      "conf": "{}/ncf_conf.json".format(save_path),
      "dsl": "{}/ncf_dsl.json".format(save_path)
    }
  }
}
    ## generate upload.json
    upload_dict={
  "file": dataset_path,
  "id_delimiter": ",",
  "head": 1,
  "partition": 4,
  "work_mode": 0,
  "backend": 0,
  "storage_engine": "PATH",
  "namespace": "experiment",
  "table_name": "recommend_matrix",
  "use_local_data": 0
}

    ## config saving...
    with open(save_path+"/ncf_conf.json",'w',encoding="utf8") as f:
        json.dump(ncf_conf,f)
    print("saving config file DONE.")
    with open(save_path+"/ncf_dsl.json",'w',encoding="utf8") as f:
        json.dump(dsl_conf,f)
    print("saving dsl config file DONE.")
    with open(save_path+"/ncf_suite.json",'w',encoding="utf8") as f:
        json.dump(suite_dict,f)
    print("saving suite config file DONE.")
    with open(save_path+"/ncf_upload.json",'w',encoding="utf8") as f:
        json.dump(upload_dict,f)
    print("saving upload config file DONE.")

    print("all things done.")
    
def setup_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=256, type=int, required=False, help="batchsize" )
    parser.add_argument("--epoch", default=2000, type=int, required=False, help="epoch")
    parser.add_argument("--lr",default=3e-5, type=float, required=False, help="learning rate")
    parser.add_argument("--dataset_path",  type=str, required=True)
    parser.add_argument("--save_path", default='../configs/', type=str, required=False)
    parser.add_argument("--feature_dim", type=int, default=200, required=True)
    return parser.parse_args()

if __name__=="__main__":
    args=setup_args()
    exportNCFPlease(transform_dim=args.feature_dim,
                    lr=args.lr,
                    dataset_path=args.dataset_path,
                    batch_size=args.batch_size,
                    save_path=args.save_path)
