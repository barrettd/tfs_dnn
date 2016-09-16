//
//  DnnTrainerNesterov.hpp
//
//  Created by Barrett Davis on 9/13/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//

#ifndef DnnTrainerNesterov_hpp
#define DnnTrainerNesterov_hpp

#include "DnnTrainer.hpp"

namespace tfs {
    
    class DnnTrainerNesterov : public DnnTrainer {

    protected:
        virtual DNN_NUMERIC adjustWeights( void );

    public:
        DnnTrainerNesterov( Dnn *dnn );
        virtual ~DnnTrainerNesterov( void );
                
    };
    
}   // namespace tfs


#endif /* DnnTrainerNesterov_hpp */
