//
//  DnnTrainerNesterov.hpp
//
//  Created by Barrett Davis on 9/13/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//

#ifndef DnnTrainerNesterov_hpp
#define DnnTrainerNesterov_hpp

#include "DnnTrainer.h"

namespace tfs {
    
    class DnnTrainerNesterov : public DnnTrainer {

    public:
        DnnTrainerNesterov( Dnn *dnn );
        virtual ~DnnTrainerNesterov( void );
        
        virtual DNN_NUMERIC train( const DNN_INTEGER expectation );     // Returns loss.
        
    };
    
}   // namespace tfs


#endif /* DnnTrainerNesterov_hpp */
