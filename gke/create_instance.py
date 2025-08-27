import requests


def create_instance():
    #POST https://container.googleapis.com/v1/projects/aixr-401704/locations/us-central1/clusters
    data ={
  "_qfn_cluster_node": {
    "name": "autopilot-_qfn_cluster_node-1",
    "network": "projects/aixr-401704/global/networks/default",
    "subnetwork": "projects/aixr-401704/regions/us-central1/subnetworks/default",
    "networkPolicy": {},
    "ipAllocationPolicy": {
      "useIpAliases": True,
      "clusterIpv4CidrBlock": "/17",
      "stackType": "IPV4"
    },
    "binaryAuthorization": {
      "evaluationMode": "DISABLED"
    },
    "autoscaling": {
      "enableNodeAutoprovisioning": True,
      "autoprovisioningNodePoolDefaults": {}
    },
    "networkConfig": {
      "enableIntraNodeVisibility": True,
      "datapathProvider": "ADVANCED_DATAPATH",
      "defaultEnablePrivateNodes": False
    },
    "authenticatorGroupsConfig": {},
    "databaseEncryption": {
      "state": "DECRYPTED"
    },
    "verticalPodAutoscaling": {
      "enabled": True
    },
    "releaseChannel": {
      "channel": "REGULAR"
    },
    "notificationConfig": {
      "pubsub": {}
    },
    "initialClusterVersion": "1.31.5-gke.1233000",
    "location": "us-central1",
    "autopilot": {
      "enabled": True
    },
    "loggingConfig": {
      "componentConfig": {
        "enableComponents": [
          "SYSTEM_COMPONENTS",
          "WORKLOADS"
        ]
      }
    },
    "monitoringConfig": {
      "componentConfig": {
        "enableComponents": [
          "SYSTEM_COMPONENTS",
          "STORAGE",
          "POD",
          "DEPLOYMENT",
          "STATEFULSET",
          "DAEMONSET",
          "HPA",
          "CADVISOR",
          "KUBELET"
        ]
      },
      "managedPrometheusConfig": {
        "enabled": True,
        "autoMonitoringConfig": {
          "scope": "NONE"
        }
      }
    },
    "nodePoolAutoConfig": {
      "resourceManagerTags": {}
    },
    "securityPostureConfig": {
      "mode": "BASIC",
      "vulnerabilityMode": "VULNERABILITY_DISABLED"
    },
    "controlPlaneEndpointsConfig": {
      "dnsEndpointConfig": {
        "allowExternalTraffic": True
      },
      "ipEndpointsConfig": {
        "enabled": True,
        "enablePublicEndpoint": True,
        "globalAccess": False,
        "authorizedNetworksConfig": {}
      }
    },
    "enterpriseConfig": {
      "desiredTier": "STANDARD"
    },
    "secretManagerConfig": {
      "enabled": False
    }
  }
}
    r = requests.post("https://container.googleapis.com/v1/projects/aixr-401704/locations/us-central1/clusters", data)
