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
    public:
        DnnTrainerSGD( Dnn *dnn );
        virtual ~DnnTrainerSGD( void );
        
        virtual DNN_NUMERIC train( const DNN_INTEGER expectation );     // Returns loss.

    };
    
}   // namespace tfs

#endif /* DnnTrainerSGD_hpp */
