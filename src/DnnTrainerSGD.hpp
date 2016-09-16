// --------------------------------------------------------------------
//  DnnTrainerSGD.hpp - Stochastic Gradient Descent
//
//  Created by Barrett Davis on 5/11/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnTrainerSGD_hpp
#define DnnTrainerSGD_hpp

#include "DnnTrainer.hpp"


namespace tfs {
    
    class DnnTrainerSGD : public DnnTrainer {
    protected:
        virtual DNN_NUMERIC adjustWeights( void );

    public:
        DnnTrainerSGD( Dnn *dnn );
        virtual ~DnnTrainerSGD( void );
        
    };
    
}   // namespace tfs

#endif /* DnnTrainerSGD_hpp */
