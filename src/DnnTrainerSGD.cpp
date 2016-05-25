// --------------------------------------------------------------------
//  DnnTrainerSGD.cpp - Stochastic Gradient Descent
//
//  Created by Barrett Davis on 5/11/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnTrainerSGD.h"
#include "Error.h"

namespace tfs {
    
    DnnTrainerSGD::DnnTrainerSGD( Dnn *dnn ) :
    tfs::DnnTrainer( dnn ) {
        // Constructor
    }
    
    DnnTrainerSGD::~DnnTrainerSGD( void ) {
        // Destructor
    }
    
    DNN_NUMERIC
    DnnTrainerSGD::train( const DNN_INTEGER expectation ) {
        // Assume: Input matrix already set for the DNN
        m_loss = 0.0;
        if( m_dnn == 0 ) {
            log_error( "No DNN set" );
            return m_loss;
        }
        if( m_batch_size == 0 ) {
            log_error( "Batch size 0 - will be a divide by zero error" );
            return m_loss;
        }
        if( !m_dnn->forward()) {
            return m_loss;
        }
        m_loss = m_dnn->backprop( expectation );
        m_k++;
        if( m_k % m_batch_size ) {
            return m_loss;
        }
        // Set up for modifying the gradiants.
        DNN_NUMERIC l1_decay_loss = 0.0;
        DNN_NUMERIC l2_decay_loss = 0.0;
        
        Trainable **trainableHandle = m_trainable_handle;
        Trainable **trainableEnd    = m_trainable_end;
        
        while( trainableHandle < trainableEnd ) {
                  Trainable   *trainable = *trainableHandle++;      // trainable != 0 & ok() from DnnTrainer::setUpTrainables()
                  DNN_NUMERIC *weight    = trainable->weightStart;
            const DNN_NUMERIC *weightEnd = trainable->weightEnd;
                  DNN_NUMERIC *gradiant  = trainable->gradiantStart;
            const DNN_NUMERIC l1_decay   = m_l1_decay * trainable->l1_decay_mul;
            const DNN_NUMERIC l2_decay   = m_l2_decay * trainable->l2_decay_mul;
            
            while( weight < weightEnd ) {
                const DNN_NUMERIC ww = *weight;
                
                l1_decay_loss += l1_decay * fabs( ww );
                l2_decay_loss += l2_decay * ww * ww / 2.0;          // accumulate weight decay loss
                
                const DNN_NUMERIC l1grad = l1_decay * (ww > 0.0 ? 1.0 : -1.0);
                const DNN_NUMERIC l2grad = l2_decay * ww;
            
                const DNN_NUMERIC gij = ( l1grad + l2grad + *gradiant++ ) / m_batch_size; // raw batch gradient

                *weight++ -= m_learning_rate * gij;
            }
        }
        return m_loss += (l1_decay_loss + l2_decay_loss);
    }

//    if(this.k % this.batch_size === 0) {
//        
//        var pglist = this.net.getParamsAndGrads();
//        
//        // initialize lists for accumulators. Will only be done once on first iteration
//        if(this.gsum.length === 0 && (this.method !== 'sgd' || this.momentum > 0.0)) {
//            // only vanilla sgd doesnt need either lists
//            // momentum needs gsum
//            // adagrad needs gsum
//            // adam and adadelta needs gsum and xsum
//            for(var i=0;i<pglist.length;i++) {
//                this.gsum.push(global.zeros(pglist[i].params.length));
//                if(this.method === 'adam' || this.method === 'adadelta') {
//                    this.xsum.push(global.zeros(pglist[i].params.length));
//                } else {
//                    this.xsum.push([]); // conserve memory
//                }
//            }
//        }
//        
//        // perform an update for all sets of weights
//        for(var i=0;i<pglist.length;i++) {
//            var pg = pglist[i]; // param, gradient, other options in future (custom learning rate etc)
//            var p = pg.params;
//            var g = pg.grads;
//            
//            // learning rate for some parameters.
//            var l2_decay_mul = typeof pg.l2_decay_mul !== 'undefined' ? pg.l2_decay_mul : 1.0;
//            var l1_decay_mul = typeof pg.l1_decay_mul !== 'undefined' ? pg.l1_decay_mul : 1.0;
//            var l2_decay = this.l2_decay * l2_decay_mul;
//            var l1_decay = this.l1_decay * l1_decay_mul;
//            
//            var plen = p.length;
//            for(var j=0;j<plen;j++) {
//                l2_decay_loss += l2_decay*p[j]*p[j]/2; // accumulate weight decay loss
//                l1_decay_loss += l1_decay*Math.abs(p[j]);
//                var l1grad = l1_decay * (p[j] > 0 ? 1 : -1);
//                var l2grad = l2_decay * (p[j]);
//                
//                var gij = (l2grad + l1grad + g[j]) / this.batch_size; // raw batch gradient
//                
//                var gsumi = this.gsum[i];
//                var xsumi = this.xsum[i];
//                if(this.method === 'adam') {
//                    // adam update
//                    gsumi[j] = gsumi[j] * this.beta1 + (1- this.beta1) * gij; // update biased first moment estimate
//                    xsumi[j] = xsumi[j] * this.beta2 + (1-this.beta2) * gij * gij; // update biased second moment estimate
//                    var biasCorr1 = gsumi[j] * (1 - Math.pow(this.beta1, this.k)); // correct bias first moment estimate
//                    var biasCorr2 = xsumi[j] * (1 - Math.pow(this.beta2, this.k)); // correct bias second moment estimate
//                    var dx =  - this.learning_rate * biasCorr1 / (Math.sqrt(biasCorr2) + this.eps);
//                    p[j] += dx;
//                } else if(this.method === 'adagrad') {
//                    // adagrad update
//                    gsumi[j] = gsumi[j] + gij * gij;
//                    var dx = - this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij;
//                    p[j] += dx;
//                } else if(this.method === 'windowgrad') {
//                    // this is adagrad but with a moving window weighted average
//                    // so the gradient is not accumulated over the entire history of the run.
//                    // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
//                    gsumi[j] = this.ro * gsumi[j] + (1-this.ro) * gij * gij;
//                    var dx = - this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij; // eps added for better conditioning
//                    p[j] += dx;
//                } else if(this.method === 'adadelta') {
//                    gsumi[j] = this.ro * gsumi[j] + (1-this.ro) * gij * gij;
//                    var dx = - Math.sqrt((xsumi[j] + this.eps)/(gsumi[j] + this.eps)) * gij;
//                    xsumi[j] = this.ro * xsumi[j] + (1-this.ro) * dx * dx; // yes, xsum lags behind gsum by 1.
//                    p[j] += dx;
//                } else if(this.method === 'nesterov') {
//                    var dx = gsumi[j];
//                    gsumi[j] = gsumi[j] * this.momentum + this.learning_rate * gij;
//                    dx = this.momentum * dx - (1.0 + this.momentum) * gsumi[j];
//                    p[j] += dx;
//                } else {
//                    // assume SGD
//                    if(this.momentum > 0.0) {
//                        // momentum update
//                        var dx = this.momentum * gsumi[j] - this.learning_rate * gij; // step
//                        gsumi[j] = dx; // back this up for next iteration of momentum
//                        p[j] += dx; // apply corrected gradient
//                    } else {
//                        // vanilla sgd
//                        p[j] +=  - this.learning_rate * gij;
//                    }
//                }
//                g[j] = 0.0; // zero out gradient so that we can begin accumulating anew
//            }
//        }
//    }
//    
//    // appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
//    // in future, TODO: have to completely redo the way loss is done around the network as currently
//    // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
//    // and it should all be computed correctly and automatically. 
//    return {fwd_time: fwd_time, bwd_time: bwd_time, 
//    l2_decay_loss: l2_decay_loss, l1_decay_loss: l1_decay_loss,
//    cost_loss: cost_loss, softmax_loss: cost_loss, 
//        loss: cost_loss + l1_decay_loss + l2_decay_loss}
//}
//}

    
}   // namespace tfs
