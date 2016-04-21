#!/bin/bash
cl run --request-docker-image tianlins/deeprl:0.0.4 :data :cbtest :script "cp script/* ./; $1"  | ./makelink
